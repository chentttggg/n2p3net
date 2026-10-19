import N2P3.PersonalRestore
import N2P3.PersonalAdmission
import Mathlib.Data.List.Sort
import N2P3.ERPFeatures

/-!
# Literal ERP family state reader and concrete personal tensor projection

Source: structured_erp.py:54-85,147-204, temporal_transfer.py:75-129,
calibration_candidates.py:311-316 and calibrated.py:618-640.
Fields are parsed unique-key mappings and Tensor values enumerate the exact
shape product. All supplied scalar values in this layer are decoded finite
reals/integers/booleans; float string conversion and nonfinite IEEE states are
outside this first same-representation domain. Numeric tensor normalization is
per element, not an assumed successful model/score roundtrip. `float64` is the
logical numeric dtype; Torch and NumPy byte/storage ownership is not proved.

The temporal reader exposes only small external source operations: constructing
PreprocessingSpec (without inventing a call to validate), numeric metadata-array
coercion, allclose orthogonality, the actual floating cosine array, participant
identity decoding and origin-policy checks. Schema/digest/role/rank/shape,
receipt structure, duplicate participants and study accounting are literal.
The xDAWN reference gate remains allclose symmetry plus eigvalsh positivity;
passing it never asserts exact symmetry or exact positive definiteness.

Exact family state keys differ from the extensible outer personal mapping.
ERP intercept must have shape [], even though saved waveform LDA permits any
numel-one intercept. ERP coefficient width is actually checked by this loader.
Fit-only covariance/shrinkage fields are not invented as restored state.
-/

namespace N2P3.ERPState

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState

noncomputable section

attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive Family where
  | pooled | toeplitz | temporal | xdawn
  deriving DecidableEq

def Family.text : Family → String
  | .pooled => "pooled_lda"
  | .toeplitz => "toeplitz_lda"
  | .temporal => "temporal_lda"
  | .xdawn => "xdawn_erp"

def readFamily : Value → Check Family
  | .text "pooled_lda" => .ok .pooled
  | .text "toeplitz_lda" => .ok .toeplitz
  | .text "temporal_lda" => .ok .temporal
  | .text "xdawn_erp" => .ok .xdawn
  | .sequence _ | .object _ => .error (.fieldFailure "unhashable ERP family")
  | _ => .error (.sourceReject "ERP family")

def positiveInt (value : Value) (stage : String) : Check ℕ :=
  match value with
  | .integer z => do
    ensure (0 < z) stage
    pure z.toNat
  | _ => .error (.sourceReject stage)

def objectAt (value : Value) (site : String) : Check Fields :=
  match value with
  | .object f => .ok f
  | _ => .error (.fieldFailure site)

def listAt (value : Value) (site : String) : Check (List Value) :=
  match value with
  | .sequence xs => .ok (valuesToList xs)
  | _ => .error (.fieldFailure site)

def normalizedValues : List Value → Check (List Value)
  | [] => .ok []
  | value :: tail => do
    let r ← match numericScalar value with
      | some r => .ok r
      | none => .error (.fieldFailure "np.asarray float64 scalar")
    let rest ← normalizedValues tail
    pure (.number r :: rest)

def normalizedTensor (tensor : Tensor) : Check Tensor := do
  let values ← normalizedValues tensor.values
  pure ⟨"float64", tensor.shape, values⟩

def normalizedState : NamedState → Check NamedState
  | [] => .ok []
  | (name, tensor) :: tail => do
    let tensor ← normalizedTensor tensor
    let rest ← normalizedState tail
    pure ((name, tensor) :: rest)

def tensorAt (state : NamedState) (name : String) : Check Tensor :=
  match lookupTensor name state with
  | some tensor => .ok tensor
  | none => .error (.fieldFailure ("missing ERP tensor:" ++ name))

def shapeAt (state : NamedState) (name : String) (shape : List ℕ) : Check Unit := do
  let tensor ← tensorAt state name
  ensure (tensor.shape = shape) "ERP state shape"

def stateKeys (state : NamedState) : List String := state.map Prod.fst

def exactStateKeys (state : NamedState) (expected : List String) : Prop :=
  stateKeys state ⊆ expected ∧ expected ⊆ stateKeys state

def sameNumericArray (a b : Tensor) : Prop := a.shape = b.shape ∧ a.values = b.values

structure PreprocessingImage where
  times : ℕ
  baseline : Value

structure IdentityImage where
  authority : String
  origins : List String
  globals : List String
  studies : List String

structure TemporalPrimitives where
  metadataHash : Fields → String
  preprocessing : Value → Check PreprocessingImage
  metadataArray : Value → Check Tensor
  orthogonalAllclose : Tensor → Bool
  cosine : ℕ → ℕ → Tensor
  identity : Value → Check IdentityImage
  originsAllowed : List String → Bool
  eventIdsUniqueHashable : Value → Check Bool

inductive Role where
  | fixed | source
  deriving DecidableEq

def readRole : Value → Check Role
  | .text "fixed_analytic" => .ok .fixed
  | .text "source_supervised" => .ok .source
  | .sequence _ | .object _ => .error (.fieldFailure "unhashable temporal fit_role")
  | _ => .error (.sourceReject "temporal schema/role")

structure TemporalModel where
  record : Fields
  matrix : Tensor
  times : ℕ
  rank : ℕ
  baseline : Value
  role : Role

def string64 : Value → Prop
  | .text s => s.toList.length = 64
  | _ => False

def readReceipt (p : TemporalPrimitives) (value : Value) : Check IdentityImage := do
  let f ← objectAt value "temporal receipt.get"
  ensure (PyTruthy (fieldDefault f "participant_key") ∧ PyTruthy (fieldDefault f "origin_keys") ∧
    PyTruthy (fieldDefault f "event_ids")) "temporal receipt header"
  let unique ← p.eventIdsUniqueHashable (fieldDefault f "event_ids")
  ensure (unique = true ∧ string64 (fieldDefault f "values_sha256") ∧
    string64 (fieldDefault f "labels_sha256") ∧ string64 (fieldDefault f "observed_sha256"))
    "temporal receipt digests/events"
  let identity ← p.identity (← required f "identity")
  ensure (p.originsAllowed identity.origins = true) "temporal origin exclusion"
  ensure (PersonalAdmission.PyEq (.text identity.authority) (← required f "participant_key") ∧
    PersonalAdmission.PyEq (.sequence (encodeTexts identity.origins)) (← required f "origin_keys") ∧
    PersonalAdmission.PyEq (.sequence (encodeTexts identity.globals)) (← required f "global_keys"))
    "temporal receipt identity"
  pure identity

def readReceipts (p : TemporalPrimitives) : List Value → Check (List IdentityImage)
  | [] => .ok []
  | value :: tail => do
    let id ← readReceipt p value
    let ids ← readReceipts p tail
    pure (id :: ids)

def canonicalStudies (identities : List IdentityImage) : List String :=
  ((identities.flatMap IdentityImage.studies).dedup).mergeSort (· ≤ ·)

def readTemporal (p : TemporalPrimitives) (raw : Value) : Check TemporalModel := by
  classical
  exact do
    let f ← objectAt raw "temporal basis record.get"
    ensure (fieldDefault f "schema" = .text "n2p3_temporal_basis/1") "temporal schema/role"
    let role ← readRole (fieldDefault f "fit_role")
    ensure (fieldDefault f "digest" = .text (p.metadataHash (f.erase "digest"))) "temporal digest"
    let preprocessing ← p.preprocessing (← required f "preprocessing")
    let rank ← positiveInt (← required f "rank") "temporal rank"
    ensure (rank ≤ preprocessing.times) "temporal rank"
    let matrix ← p.metadataArray (← required f "basis")
    ensure (matrix.shape = [preprocessing.times, rank] ∧ p.orthogonalAllclose matrix = true)
      "temporal matrix shape/finite/allclose"
    let receipts ← match fieldDefault f "fit_receipts" with
      | .sequence xs => .ok (valuesToList xs)
      | _ => .error (.sourceReject "temporal receipts list")
    match role with
    | .fixed =>
      ensure (receipts = [] ∧ ¬PyTruthy (fieldDefault f "source_studies")) "fixed basis source claims"
      ensure (sameNumericArray matrix (p.cosine preprocessing.times rank)) "fixed cosine array equality"
    | .source =>
      ensure (receipts ≠ []) "temporal receipts nonempty"
      let identities ← readReceipts p receipts
      ensure ((identities.map IdentityImage.authority).Nodup) "temporal duplicate participant"
      ensure (PersonalAdmission.PyEq (.sequence (encodeTexts (canonicalStudies identities)))
        (fieldDefault f "source_studies")) "temporal source studies"
    pure ⟨f, matrix, preprocessing.times, rank, preprocessing.baseline, role⟩

def optionalTemporal (p : TemporalPrimitives) (record : Fields) : Check (Option TemporalModel) :=
  match record.get "temporal_basis" with
  | none => .ok none
  | some value => some <$> readTemporal p value

structure Header where
  family : Family
  binWidth : ℕ
  nfilter : ℕ
  baseline : Value
  temporal : Option TemporalModel

def commonHeader (p : TemporalPrimitives) (record : Fields)
    (temporal : Option TemporalModel) : Check Header := by
  classical
  exact do
    let familyValue ← required record "family"
    let widthValue ← required record "bin_width"
    let baseline ← required record "baseline_mode"
    let nfilterValue ← required record "nfilter"
    let family ← readFamily familyValue
    let width ← positiveInt widthValue "ERP bin_width"
    ensure (family ≠ .toeplitz ∨ baseline = .text "none") "Toeplitz baseline"
    let nfilter ← positiveInt nfilterValue "ERP nfilter"
    match family, temporal with
    | .temporal, some basis =>
      ensure (width = 1) "temporal ERP basis/bin"
      let basis ← readTemporal p (.object basis.record)
      ensure (PersonalAdmission.PyEq basis.baseline baseline) "temporal ERP baseline"
      pure ⟨family, width, nfilter, baseline, some basis⟩
    | .temporal, none => .error (.sourceReject "temporal ERP basis/bin")
    | _, some _ => .error (.sourceReject "temporal basis family")
    | _, none => pure ⟨family, width, nfilter, baseline, none⟩

def readGeometry (record : Fields) : Check (ℕ × ℕ) := do
  let value ← required record "input_shape"
  match value with
  | .sequence (.cons (.integer c) (.cons (.integer t) .nil)) => do
    ensure (0 < c ∧ 0 < t) "ERP input geometry"
    pure (c.toNat, t.toNat)
  | .null | .boolean _ | .integer _ | .number _ => .error (.fieldFailure "tuple(input_shape)")
  | _ => .error (.sourceReject "ERP input geometry")

def expectedKeys (header : Header) : List String :=
  ["coefficient", "intercept"] ++
    (if header.family = .xdawn then ["filters", "prototypes", "reference"] else []) ++
    (if header.temporal.isSome then ["temporal_basis"] else [])

structure ReferencePrimitives where
  symmetryAllclose : Tensor → Bool
  eigenvaluesPositive : Tensor → Bool

structure Model where
  header : Header
  channels : ℕ
  times : ℕ
  state : NamedState
  observedRank : Option ℕ
  effectiveNfilter : Option ℕ

def featureWidth (header : Header) (channels times : ℕ) : ℕ :=
  channels * (match header.temporal with
    | none => times / header.binWidth
    | some temporal => temporal.rank)

def fromState (p : TemporalPrimitives) (reference : ReferencePrimitives)
    (record : Fields) (state : NamedState) : Check Model := by
  classical
  exact do
    ensure (fieldDefault record "schema" = .text "n2p3_structured_erp/1" ∧
      fieldDefault record "feature_order" = .text "channel_prime") "ERP schema/order"
    let temporal ← optionalTemporal p record
    let header ← commonHeader p record temporal
    let (c, t) ← readGeometry record
    let bins := t / header.binWidth
    ensure (0 < bins ∧ PersonalAdmission.PyEq (fieldDefault record "discarded_tail_samples")
      (.integer (t % header.binWidth))) "ERP binning"
    ensure (exactStateKeys state (expectedKeys header)) "ERP state keys"
    let state ← normalizedState state
    match header.family with
    | .xdawn =>
      let rank ← positiveInt (fieldDefault record "observed_rank") "xDAWN rank/effective"
      let effective ← positiveInt (fieldDefault record "effective_nfilter") "xDAWN rank/effective"
      ensure (rank ≤ c ∧ effective = min header.nfilter rank) "xDAWN rank/effective"
      let v := 2 * effective
      let covariance ← tensorAt state "reference"
      if covariance.shape = [2 * v, 2 * v] then
        ensure (reference.symmetryAllclose covariance = true ∧ reference.eigenvaluesPositive covariance = true)
          "xDAWN reference symmetry/positive eigenvalues"
      else pure ()
      shapeAt state "filters" [v, c]
      shapeAt state "prototypes" [v, bins]
      shapeAt state "reference" [2 * v, 2 * v]
      shapeAt state "coefficient" [(2 * v) * (2 * v + 1) / 2]
      shapeAt state "intercept" []
      pure ⟨header, c, t, state, some rank, some effective⟩
    | _ =>
      match header.temporal with
      | some temporal =>
        let matrix ← tensorAt state "temporal_basis"
        ensure (temporal.times = t ∧ sameNumericArray matrix temporal.matrix) "temporal duplicate basis equality"
      | none => pure ()
      shapeAt state "coefficient" [featureWidth header c t]
      shapeAt state "intercept" []
      match header.temporal with
      | some temporal => shapeAt state "temporal_basis" temporal.matrix.shape
      | none => pure ()
      pure ⟨header, c, t, state, none, none⟩

def copyState : NamedState → NamedState
  | [] => []
  | (name, tensor) :: tail => (name, ⟨tensor.dtype, tensor.shape, tensor.values.map copyValue⟩) :: copyState tail

def exportRecord (model : Model) : Fields := Fields.ofList
  ([("schema", .text "n2p3_structured_erp/1"), ("family", .text model.header.family.text),
    ("bin_width", .integer model.header.binWidth), ("baseline_mode", model.header.baseline),
    ("nfilter", .integer model.header.nfilter),
    ("input_shape", .sequence (encodeNats [model.channels, model.times])),
    ("feature_order", .text "channel_prime"),
    ("discarded_tail_samples", .integer (model.times % model.header.binWidth))] ++
   (match model.header.temporal with | none => [] | some basis => [("temporal_basis", .object basis.record)]) ++
   (match model.observedRank, model.effectiveNfilter with
    | some rank, some effective => [("observed_rank", .integer rank), ("effective_nfilter", .integer effective)]
    | _, _ => []))

def exportState (model : Model) : Fields × NamedState := ⟨exportRecord model, copyState model.state⟩

/-- Exactly the personal source's startswith/removeprefix comprehension. -/
def stripERP (name : String) : Option String :=
  match name.toList with
  | 'e' :: 'r' :: 'p' :: '.' :: tail => some (String.mk tail)
  | _ => none

def erpFields : NamedState → NamedState
  | [] => []
  | (name, tensor) :: tail =>
    match stripERP name with
    | some suffix => (suffix, tensor) :: erpFields tail
    | none => erpFields tail

def writePersonalERP (model : Model) (channels : Tensor) (outerExtras : NamedState := []) : NamedState :=
  (copyState model.state).map (fun entry => ("erp." ++ entry.1, entry.2)) ++
    ("waveform_channels", channels) :: outerExtras

def realTensor (shape : List ℕ) (values : List ℝ) : Tensor :=
  ⟨"float64", shape, values.map Value.number⟩

def coefficientTensor {d : ℕ} (coefficient : Fin d → ℝ) : Tensor :=
  realTensor [d] (List.ofFn coefficient)

def interceptTensor (intercept : ℝ) : Tensor := realTensor [] [intercept]

def strictScalar (tensor : Tensor) : Option ℝ :=
  if tensor.shape = [] then
    match tensor.values with | [value] => numericScalar value | _ => none
  else none

def linearState {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ) : NamedState :=
  [("coefficient", coefficientTensor coefficient), ("intercept", interceptTensor intercept)]

def plainRecord (family : Family) (width nfilter channels times : ℕ) (baseline : Value) : Fields := Fields.ofList
  [("schema", .text "n2p3_structured_erp/1"), ("family", .text family.text),
   ("bin_width", .integer width), ("baseline_mode", baseline), ("nfilter", .integer nfilter),
   ("input_shape", .sequence (encodeNats [channels, times])),
   ("feature_order", .text "channel_prime"), ("discarded_tail_samples", .integer (times % width))]

theorem normalized_real_values (values : List ℝ) :
    normalizedValues (values.map Value.number) = .ok (values.map Value.number) := by
  induction values with
  | nil => rfl
  | cons value tail ih => simp [normalizedValues, numericScalar, ih]

theorem normalized_real_ofFn {d : ℕ} (values : Fin d → ℝ) :
    normalizedValues (List.ofFn (Value.number ∘ values)) =
      .ok (List.ofFn (Value.number ∘ values)) := by
  rw [← List.map_ofFn]
  exact normalized_real_values _

theorem normalized_real_tensor (shape : List ℕ) (values : List ℝ) :
    normalizedTensor (realTensor shape values) = .ok (realTensor shape values) := by
  simp [normalizedTensor, realTensor, normalized_real_values]

theorem normalized_linear_state {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ) :
    normalizedState (linearState coefficient intercept) = .ok (linearState coefficient intercept) := by
  simp [linearState, normalizedState, coefficientTensor, interceptTensor, normalized_real_tensor]

theorem scalar_rank_is_strict (value : ℝ) :
    strictScalar (realTensor [1] [value]) = none ∧ strictScalar (interceptTensor value) = some value := by
  simp [strictScalar, realTensor, interceptTensor, numericScalar]

theorem read_positive_integer (n : ℕ) (stage : String) (h : 0 < n) :
    positiveInt (.integer n) stage = .ok n := by
  simp [positiveInt, ensure, h]

theorem family_text_decodes (family : Family) : readFamily (.text family.text) = .ok family := by
  cases family <;> rfl

theorem plain_common_header (p : TemporalPrimitives) (family : Family) (width nfilter c t : ℕ)
    (baseline : Value) (hw : 0 < width) (hn : 0 < nfilter)
    (ht : family ≠ .temporal) (hb : family ≠ .toeplitz ∨ baseline = .text "none") :
    commonHeader p (plainRecord family width nfilter c t baseline) none =
      .ok ⟨family, width, nfilter, baseline, none⟩ := by
  cases family <;> simp_all [commonHeader, plainRecord, Fields.ofList, Fields.get, required,
    Family.text, readFamily, positiveInt, ensure]

theorem plain_family_constructs (p : TemporalPrimitives) (reference : ReferencePrimitives)
    (family : Family) (width nfilter c t : ℕ) (baseline : Value)
    (coefficient : Fin (c * (t / width)) → ℝ) (intercept : ℝ)
    (hw : 0 < width) (hn : 0 < nfilter) (hc : 0 < c) (ht : 0 < t) (hb : 0 < t / width)
    (hf : family ≠ .temporal ∧ family ≠ .xdawn)
    (hbaseline : family ≠ .toeplitz ∨ baseline = .text "none") :
    fromState p reference (plainRecord family width nfilter c t baseline) (linearState coefficient intercept) =
      .ok ⟨⟨family, width, nfilter, baseline, none⟩, c, t, linearState coefficient intercept, none, none⟩ := by
  have hh := plain_common_header p family width nfilter c t baseline hw hn hf.1 hbaseline
  cases family <;> simp_all [fromState, plainRecord, fieldDefault, Fields.ofList, Fields.get,
    optionalTemporal, readGeometry, encodeNats, required, ensure, exactStateKeys, stateKeys,
    expectedKeys, linearState, normalizedState, coefficientTensor, interceptTensor,
    normalizedTensor, normalizedValues, numericScalar, normalized_real_ofFn,
    shapeAt, tensorAt, lookupTensor, realTensor, featureWidth,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]

theorem copied_state_preserves_all_fields (state : NamedState) : copyState state = state := by
  induction state with
  | nil => rfl
  | cons entry tail ih =>
    rcases entry with ⟨name, tensor⟩
    have hv : tensor.values.map copyValue = tensor.values := by
      induction tensor.values with
      | nil => rfl
      | cons value tail ih => simp [copyValue_preserves, ih]
    simp [copyState, hv, ih]

theorem exported_numeric_fields_are_preserved (model : Model) : (exportState model).2 = model.state :=
  copied_state_preserves_all_fields model.state

theorem vector_field_reads_exactly {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ) :
    (tensorAt (linearState coefficient intercept) "coefficient").map
      (PersonalRestore.tensorVector d) = .ok (some coefficient) := by
  simp [tensorAt, linearState, lookupTensor, PersonalRestore.tensorVector, coefficientTensor,
    realTensor, List.map_ofFn, Function.comp_def, PersonalRestore.decodeVector_ofFn]

theorem intercept_field_reads_exactly {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ) :
    (tensorAt (linearState coefficient intercept) "intercept").map strictScalar = .ok (some intercept) := by
  simp [tensorAt, linearState, lookupTensor, strictScalar, interceptTensor, realTensor, numericScalar]

def affineScore (state : NamedState) (d : ℕ) (features : Fin d → ℝ) : Check ℝ := do
  let tensor ← tensorAt state "coefficient"
  let coefficient ← match PersonalRestore.tensorVector d tensor with
    | some coefficient => .ok coefficient
    | none => .error (.sourceReject "ERP coefficient decode")
  let tensor ← tensorAt state "intercept"
  let intercept ← match strictScalar tensor with
    | some intercept => .ok intercept
    | none => .error (.sourceReject "ERP scalar intercept decode")
  pure (dotProduct features coefficient + intercept)

theorem affine_score_from_independent_fields {d : ℕ} (coefficient : Fin d → ℝ)
    (intercept : ℝ) (features : Fin d → ℝ) :
    affineScore (linearState coefficient intercept) d features =
      .ok (dotProduct features coefficient + intercept) := by
  simp [affineScore, tensorAt, linearState, lookupTensor, PersonalRestore.tensorVector,
    coefficientTensor, interceptTensor, realTensor, strictScalar, numericScalar,
    List.map_ofFn, Function.comp_def, PersonalRestore.decodeVector_ofFn]

theorem prefixed_name_strips_once (name : String) : stripERP ("erp." ++ name) = some name := by
  cases name
  rfl

theorem prefixed_state_reads_back (state : NamedState) :
    erpFields (state.map (fun entry => ("erp." ++ entry.1, entry.2))) = state := by
  induction state with
  | nil => rfl
  | cons entry tail ih =>
    rcases entry with ⟨name, tensor⟩
    simp [erpFields, prefixed_name_strips_once, ih]

theorem erp_fields_append (first second : NamedState) :
    erpFields (first ++ second) = erpFields first ++ erpFields second := by
  induction first with
  | nil => rfl
  | cons entry tail ih =>
    rcases entry with ⟨name, tensor⟩
    cases h : stripERP name <;> simp [erpFields, h, ih]

def NoERPExtras (extras : NamedState) : Prop := erpFields extras = []

theorem personal_prefix_parser_recovers_fields (model : Model) (channels : Tensor)
    (extras : NamedState) (h : NoERPExtras extras) :
    erpFields (writePersonalERP model channels extras) = model.state := by
  rw [writePersonalERP, erp_fields_append, prefixed_state_reads_back, copied_state_preserves_all_fields]
  simpa [erpFields, stripERP, NoERPExtras] using h

structure FixedTemporalInput where
  preprocessing : Value
  rawBasis : Value
  times : ℕ
  rank : ℕ
  baseline : Value
  matrix : Tensor

def fixedTemporalMetadata (input : FixedTemporalInput) : Fields := Fields.ofList
  [("schema", .text "n2p3_temporal_basis/1"), ("fit_role", .text "fixed_analytic"),
   ("preprocessing", input.preprocessing), ("rank", .integer input.rank), ("basis", input.rawBasis),
   ("fit_receipts", .sequence .nil), ("source_studies", .sequence .nil), ("risk", .text "none")]

def fixedTemporalRecord (p : TemporalPrimitives) (input : FixedTemporalInput) : Fields :=
  .cons "digest" (.text (p.metadataHash (fixedTemporalMetadata input))) (fixedTemporalMetadata input)

def fixedTemporalModel (p : TemporalPrimitives) (input : FixedTemporalInput) : TemporalModel :=
  ⟨fixedTemporalRecord p input, input.matrix, input.times, input.rank, input.baseline, .fixed⟩

/-- Each premise concerns an actual small operation or explicit source guard. -/
def FixedTemporalConditions (p : TemporalPrimitives) (input : FixedTemporalInput) : Prop :=
  p.preprocessing input.preprocessing = .ok ⟨input.times, input.baseline⟩ ∧
  0 < input.rank ∧ input.rank ≤ input.times ∧
  p.metadataArray input.rawBasis = .ok input.matrix ∧
  input.matrix.shape = [input.times, input.rank] ∧ p.orthogonalAllclose input.matrix = true ∧
  sameNumericArray input.matrix (p.cosine input.times input.rank)

theorem fixed_temporal_reader_from_explicit_fields (p : TemporalPrimitives) (input : FixedTemporalInput)
    (h : FixedTemporalConditions p input) :
    readTemporal p (.object (fixedTemporalRecord p input)) = .ok (fixedTemporalModel p input) := by
  rcases h with ⟨hpre, hrpos, hrmax, hbasis, hshape, horth, hcos⟩
  simp [readTemporal, fixedTemporalRecord, fixedTemporalMetadata, objectAt, fieldDefault,
    Fields.ofList, Fields.get, Fields.erase, required, ensure, valuesToList,
    readRole, positiveInt, PyTruthy, hpre, hrpos, hrmax, hbasis, hshape, horth, hcos, fixedTemporalModel]

def temporalERPRecord (p : TemporalPrimitives) (basis : FixedTemporalInput)
    (nfilter c : ℕ) : Fields := .cons "temporal_basis" (.object (fixedTemporalRecord p basis))
      (plainRecord .temporal 1 nfilter c basis.times basis.baseline)

def temporalNumericState {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ)
    (matrix : Tensor) : NamedState := linearState coefficient intercept ++ [("temporal_basis", matrix)]

theorem normalized_temporal_state {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ)
    (shape : List ℕ) (values : List ℝ) :
    normalizedState (temporalNumericState coefficient intercept (realTensor shape values)) =
      .ok (temporalNumericState coefficient intercept (realTensor shape values)) := by
  simp [temporalNumericState, linearState, normalizedState, coefficientTensor, interceptTensor,
    normalized_real_tensor]

theorem temporal_common_header (p : TemporalPrimitives) (basis : FixedTemporalInput)
    (nfilter c : ℕ) (h : FixedTemporalConditions p basis) (hn : 0 < nfilter) :
    commonHeader p (temporalERPRecord p basis nfilter c) (some (fixedTemporalModel p basis)) =
      .ok ⟨.temporal, 1, nfilter, basis.baseline, some (fixedTemporalModel p basis)⟩ := by
  have ht := fixed_temporal_reader_from_explicit_fields p basis h
  simp [commonHeader, temporalERPRecord, plainRecord, Fields.ofList, Fields.get, required,
    Family.text, readFamily, positiveInt, ensure, hn, fixedTemporalModel, ht,
    PersonalAdmission.PyEq]

theorem temporal_family_constructs (p : TemporalPrimitives) (reference : ReferencePrimitives)
    (basis : FixedTemporalInput) (nfilter c : ℕ)
    (coefficient : Fin (c * basis.rank) → ℝ) (intercept : ℝ) (values : List ℝ)
    (h : FixedTemporalConditions p basis) (hn : 0 < nfilter) (hc : 0 < c)
    (_hvalues : values.length = basis.times * basis.rank)
    (hmatrix : basis.matrix = realTensor [basis.times, basis.rank] values) :
    fromState p reference (temporalERPRecord p basis nfilter c)
      (temporalNumericState coefficient intercept basis.matrix) =
      .ok ⟨⟨.temporal, 1, nfilter, basis.baseline, some (fixedTemporalModel p basis)⟩,
        c, basis.times, temporalNumericState coefficient intercept basis.matrix, none, none⟩ := by
  have ht := fixed_temporal_reader_from_explicit_fields p basis h
  have hh := temporal_common_header p basis nfilter c h hn
  have htimes : 0 < basis.times := lt_of_lt_of_le h.2.1 h.2.2.1
  have hs : fieldDefault (temporalERPRecord p basis nfilter c) "schema" = .text "n2p3_structured_erp/1" := by
    simp [temporalERPRecord, plainRecord, fieldDefault, Fields.ofList, Fields.get]
  have ho : fieldDefault (temporalERPRecord p basis nfilter c) "feature_order" = .text "channel_prime" := by
    simp [temporalERPRecord, plainRecord, fieldDefault, Fields.ofList, Fields.get]
  have hopt : optionalTemporal p (temporalERPRecord p basis nfilter c) = .ok (some (fixedTemporalModel p basis)) := by
    simp [optionalTemporal, temporalERPRecord, Fields.get, ht]
  have hg : readGeometry (temporalERPRecord p basis nfilter c) = .ok (c, basis.times) := by
    simp [readGeometry, temporalERPRecord, plainRecord, required, Fields.ofList, Fields.get, encodeNats, ensure, hc, htimes]
  have hnorm : normalizedState (temporalNumericState coefficient intercept basis.matrix) =
      .ok (temporalNumericState coefficient intercept basis.matrix) := by
    rw [hmatrix]
    exact normalized_temporal_state _ _ _ _
  simp only [fromState, hs, ho]
  simp only [ensure, hopt, hh, hg, hnorm, Bind.bind, Except.bind, Pure.pure, Except.pure]
  simp [temporalERPRecord, plainRecord, fieldDefault, Fields.ofList, Fields.get,
    ensure, htimes,
    exactStateKeys, expectedKeys, stateKeys, temporalNumericState, linearState,
    coefficientTensor, interceptTensor, hmatrix, realTensor,
    tensorAt, lookupTensor, shapeAt, featureWidth, fixedTemporalModel, sameNumericArray,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]

def xdawnRecord (width nfilter c t rank effective : ℕ) (baseline : Value) : Fields :=
  .cons "observed_rank" (.integer rank) (.cons "effective_nfilter" (.integer effective)
    (plainRecord .xdawn width nfilter c t baseline))

def xdawnNumericState (c bins effective : ℕ)
    (filters : List ℝ) (prototypes : List ℝ) (reference : List ℝ)
    (coefficient : Fin ((4 * effective) * (4 * effective + 1) / 2) → ℝ) (intercept : ℝ) : NamedState :=
  [("coefficient", coefficientTensor coefficient), ("intercept", interceptTensor intercept),
   ("filters", realTensor [2 * effective, c] filters),
   ("prototypes", realTensor [2 * effective, bins] prototypes),
   ("reference", realTensor [4 * effective, 4 * effective] reference)]

theorem xdawn_family_constructs (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (width nfilter c t rank effective : ℕ) (baseline : Value)
    (filters prototypes reference : List ℝ)
    (coefficient : Fin ((4 * effective) * (4 * effective + 1) / 2) → ℝ) (intercept : ℝ)
    (hw : 0 < width) (hn : 0 < nfilter) (hc : 0 < c) (ht : 0 < t) (hb : 0 < t / width)
    (hr : 0 < rank ∧ rank ≤ c) (he : 0 < effective ∧ effective = min nfilter rank)
    (_hfilters : filters.length = (2 * effective) * c)
    (_hprototypes : prototypes.length = (2 * effective) * (t / width))
    (_hreference : reference.length = (4 * effective) * (4 * effective))
    (hclose : ref.symmetryAllclose (realTensor [4 * effective, 4 * effective] reference) = true)
    (hpositive : ref.eigenvaluesPositive (realTensor [4 * effective, 4 * effective] reference) = true) :
    fromState p ref (xdawnRecord width nfilter c t rank effective baseline)
      (xdawnNumericState c (t / width) effective filters prototypes reference coefficient intercept) =
      .ok ⟨⟨.xdawn, width, nfilter, baseline, none⟩, c, t,
        xdawnNumericState c (t / width) effective filters prototypes reference coefficient intercept,
        some rank, some effective⟩ := by
  have hcount : rank ≤ c ∧ effective = min nfilter rank := ⟨hr.2, he.2⟩
  dsimp [realTensor] at hclose hpositive
  have hmin : (min (nfilter : ℤ) (rank : ℤ)).toNat = min nfilter rank := by omega
  have hfactor : 2 * (2 * min nfilter rank) = 4 * min nfilter rank := by omega
  have hclose' := hclose
  have hpositive' := hpositive
  rw [he.2] at hclose' hpositive'
  simp [fromState, xdawnRecord, plainRecord, fieldDefault, Fields.ofList, Fields.get,
    optionalTemporal, commonHeader, readFamily, Family.text, positiveInt, ensure,
    hw, hn, hc, ht, hb, hr.1, hcount, readGeometry, required, encodeNats,
    exactStateKeys, expectedKeys, stateKeys, xdawnNumericState, normalizedState,
    coefficientTensor, interceptTensor, normalizedTensor, normalizedValues, numericScalar,
    normalized_real_values, normalized_real_ofFn, tensorAt, lookupTensor, realTensor,
    shapeAt, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]
  simp [hmin, hfactor, hclose', hpositive']

theorem fromState_record_lookup_ext (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (left right : Fields) (state : NamedState) (h : ∀ key, left.get key = right.get key) :
    fromState p ref left state = fromState p ref right state := by
  simp only [fromState, optionalTemporal, commonHeader, readGeometry, required, fieldDefault]
  simp only [h]

theorem loaded_plain_uses_actual_bin_features (p : TemporalPrimitives) (reference : ReferencePrimitives)
    (family : Family) (width nfilter c t : ℕ) (baseline : Value)
    (coefficient : Fin (c * (t / width)) → ℝ) (intercept : ℝ) (x : ERPFeatures.Epoch c t)
    (hw : 0 < width) (hn : 0 < nfilter) (hc : 0 < c) (ht : 0 < t) (hb : 0 < t / width)
    (hf : family ≠ .temporal ∧ family ≠ .xdawn)
    (hbaseline : family ≠ .toeplitz ∨ baseline = .text "none") :
    (fromState p reference (plainRecord family width nfilter c t baseline) (linearState coefficient intercept)).bind
      (fun model => affineScore model.state (c * (t / width)) (ERPFeatures.binFeatures width x)) =
      .ok (ERPFeatures.evaluate ⟨coefficient, intercept⟩ (ERPFeatures.binFeatures width x)) := by
  rw [plain_family_constructs p reference family width nfilter c t baseline coefficient intercept hw hn hc ht hb hf hbaseline]
  simpa only [Bind.bind, Except.bind, ERPFeatures.evaluate, dotProduct] using
    affine_score_from_independent_fields coefficient intercept (ERPFeatures.binFeatures width x)

theorem plain_export_load_replay (p : TemporalPrimitives) (reference : ReferencePrimitives)
    (family : Family) (width nfilter c t : ℕ) (baseline : Value)
    (coefficient : Fin (c * (t / width)) → ℝ) (intercept : ℝ)
    (hw : 0 < width) (hn : 0 < nfilter) (hc : 0 < c) (ht : 0 < t) (hb : 0 < t / width)
    (hf : family ≠ .temporal ∧ family ≠ .xdawn)
    (hbaseline : family ≠ .toeplitz ∨ baseline = .text "none") :
    let model : Model := ⟨⟨family, width, nfilter, baseline, none⟩, c, t,
      linearState coefficient intercept, none, none⟩
    fromState p reference (exportState model).1 (exportState model).2 = .ok model := by
  dsimp only
  rw [exported_numeric_fields_are_preserved]
  exact plain_family_constructs p reference family width nfilter c t baseline coefficient intercept hw hn hc ht hb hf hbaseline

def personalInput (model : Model) (selected : Tensor) (extras : NamedState)
    (metadataExtras : Fields) : ConstructorInput :=
  ⟨.object (.cons "erp_artifact" (.object (exportRecord model)) metadataExtras),
    writePersonalERP model selected extras⟩

/-- The numeric ERP subroute of a personal constructor, after the independent
selected-channel/QC/physical/recipe checks owned by the personal caller. This
helper does not claim to replace those source gates. -/
def restoredERPSubmodel (p : TemporalPrimitives) (reference : ReferencePrimitives)
    (mh : PersonalState.MetadataHash) (sh : StateHash) (raw : Value) : Check Model := do
  let input ← loadInput mh sh raw
  let metadata ← objectAt input.metadata "personal metadata.get"
  let record ← objectAt (← required metadata "erp_artifact") "ERP record.get"
  fromState p reference record (erpFields input.state)

theorem personal_envelope_feeds_actual_erp_reader (p : TemporalPrimitives) (reference : ReferencePrimitives)
    (mh : PersonalState.MetadataHash) (sh : StateHash) (model : Model) (selected : Tensor)
    (extras : NamedState) (metadataExtras : Fields) (h : NoERPExtras extras) :
    restoredERPSubmodel p reference mh sh
      (saveView mh sh (personalInput model selected extras metadataExtras)) =
      fromState p reference (exportRecord model) model.state := by
  simp [restoredERPSubmodel, personalInput, current_source_copy_restore, objectAt,
    required, Fields.get, personal_prefix_parser_recovers_fields model selected extras h]

def fixtureBasis : Tensor := realTensor [1, 1] [1]
def fixturePreprocessing : Value := .object (Fields.ofList
  [("n_times", .integer 1), ("baseline_mode", .text "none")])
def fixtureRawBasis : Value := .sequence (.cons (.sequence (.cons (.number 1) .nil)) .nil)

/-- Only the demonstrated preprocessing/basis/cosine operations are consumed
by the fixed 1-by-1 witness. Identity/receipt branches are never invoked. -/
def fixturePrimitives : TemporalPrimitives :=
  ⟨fun _ => "fixture-digest", fun _ => .ok ⟨1, .text "none"⟩,
    fun _ => .ok fixtureBasis, fun _ => true, fun _ _ => fixtureBasis,
    fun _ => .error (.fieldFailure "unused identity primitive"), fun _ => true,
    fun _ => .error (.fieldFailure "unused event primitive")⟩

def fixtureReference : ReferencePrimitives := ⟨fun _ => true, fun _ => true⟩
def fixtureFixedInput : FixedTemporalInput :=
  ⟨fixturePreprocessing, fixtureRawBasis, 1, 1, .text "none", fixtureBasis⟩
def fixtureCoefficient : Fin 1 → ℝ := fun _ => 2
def fixtureLinearState : NamedState := linearState fixtureCoefficient (-1)

theorem fixture_fixed_conditions : FixedTemporalConditions fixturePrimitives fixtureFixedInput := by
  exact ⟨rfl, by decide, by decide, rfl, rfl, rfl, ⟨rfl, rfl⟩⟩

def fixturePlain (family : Family) : Model :=
  ⟨⟨family, 1, 1, .text "none", none⟩, 1, 1, fixtureLinearState, none, none⟩

theorem pooled_and_toeplitz_nonempty_construct (family : Family) (hfamily : family = .pooled ∨ family = .toeplitz) :
    fromState fixturePrimitives fixtureReference (plainRecord family 1 1 1 1 (.text "none")) fixtureLinearState =
      .ok (fixturePlain family) := by
  apply plain_family_constructs
  all_goals rcases hfamily with rfl | rfl <;> simp

def fixtureTemporalState : NamedState := temporalNumericState fixtureCoefficient (-1) fixtureBasis

theorem temporal_nonempty_construct :
    fromState fixturePrimitives fixtureReference (temporalERPRecord fixturePrimitives fixtureFixedInput 1 1)
      fixtureTemporalState = .ok
      ⟨⟨.temporal, 1, 1, .text "none", some (fixedTemporalModel fixturePrimitives fixtureFixedInput)⟩,
        1, 1, fixtureTemporalState, none, none⟩ := by
  exact temporal_family_constructs fixturePrimitives fixtureReference fixtureFixedInput 1 1
    fixtureCoefficient (-1) [1] fixture_fixed_conditions (by decide) (by decide) rfl rfl

def fixtureXDawnState : NamedState := xdawnNumericState 1 1 1 [1, 2] [0, 1]
  [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] (fun _ => 1) 0

theorem xdawn_nonempty_construct :
    fromState fixturePrimitives fixtureReference (xdawnRecord 1 1 1 1 1 1 (.text "none")) fixtureXDawnState =
      .ok ⟨⟨.xdawn, 1, 1, .text "none", none⟩, 1, 1, fixtureXDawnState, some 1, some 1⟩ := by
  exact xdawn_family_constructs fixturePrimitives fixtureReference 1 1 1 1 1 1 (.text "none")
    [1, 2] [0, 1] [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] (fun _ => 1) 0
    (by decide) (by decide) (by decide) (by decide) (by decide) ⟨by decide, by decide⟩
    ⟨by decide, by decide⟩ rfl rfl rfl rfl rfl

theorem legal_prefix_restore_scores_five (mh : PersonalState.MetadataHash) (sh : StateHash) :
    (restoredERPSubmodel fixturePrimitives fixtureReference mh sh
      (saveView mh sh (personalInput (fixturePlain .pooled)
        ⟨"torch.int64", [1], [.integer 0]⟩ [("unused", realTensor [1] [9])] .nil))).bind
      (fun model => affineScore model.state 1 (fun _ => 3)) = .ok 5 := by
  rw [personal_envelope_feeds_actual_erp_reader]
  · change (fromState fixturePrimitives fixtureReference (plainRecord .pooled 1 1 1 1 (.text "none"))
      fixtureLinearState).bind _ = .ok 5
    rw [pooled_and_toeplitz_nonempty_construct .pooled (Or.inl rfl)]
    simp [fixturePlain, fixtureLinearState, affine_score_from_independent_fields, fixtureCoefficient, dotProduct]
    norm_num
  · rfl

def matrixTensor {rows cols : ℕ} (matrix : Matrix (Fin rows) (Fin cols) ℝ) : Tensor :=
  realTensor [rows, cols] (List.ofFn (fun flat : Fin (rows * cols) =>
    let pair := finProdFinEquiv.symm flat
    matrix pair.1 pair.2))

def matrixField (rows cols : ℕ) (tensor : Tensor) : Option (Matrix (Fin rows) (Fin cols) ℝ) := do
  if tensor.shape = [rows, cols] then
    let vector ← PersonalRestore.decodeVector (rows * cols) tensor.values
    pure (fun i j => vector (finProdFinEquiv (i, j)))
  else none

theorem matrix_field_reads_all_entries {rows cols : ℕ} (matrix : Matrix (Fin rows) (Fin cols) ℝ) :
    matrixField rows cols (matrixTensor matrix) = some matrix := by
  have hd : PersonalRestore.decodeVector (rows * cols) (matrixTensor matrix).values =
      some (fun flat => let pair := finProdFinEquiv.symm flat; matrix pair.1 pair.2) := by
    simp [matrixTensor, realTensor, List.map_ofFn, Function.comp_def, PersonalRestore.decodeVector_ofFn]
  unfold matrixField
  have hshape : (matrixTensor matrix).shape = [rows, cols] := rfl
  rw [if_pos hshape, hd]
  change some (fun i j => matrix (finProdFinEquiv.symm (finProdFinEquiv (i, j))).1
    (finProdFinEquiv.symm (finProdFinEquiv (i, j))).2) = some matrix
  apply congrArg some
  funext i j
  rw [Equiv.symm_apply_apply]

theorem affine_score_with_declared_tail {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ)
    (tail : NamedState) (features : Fin d → ℝ) :
    affineScore (linearState coefficient intercept ++ tail) d features =
      .ok (ERPFeatures.evaluate ⟨coefficient, intercept⟩ features) := by
  simp [affineScore, tensorAt, linearState, lookupTensor, PersonalRestore.tensorVector,
    coefficientTensor, interceptTensor, realTensor, strictScalar, numericScalar,
    List.map_ofFn, Function.comp_def, PersonalRestore.decodeVector_ofFn, ERPFeatures.evaluate, dotProduct]

/-- Prediction reads the metadata TemporalBasis matrix, not the duplicate
state tensor; the source constructor has checked their exact array equality. -/
def temporalAffineScore (c t rank : ℕ) (model : Model) (x : ERPFeatures.Epoch c t) : Check ℝ := do
  let temporal ← match model.header.temporal with
    | some temporal => .ok temporal
    | none => .error (.fieldFailure "missing temporal ERP basis")
  let matrix ← match matrixField t rank temporal.matrix with
    | some matrix => .ok matrix
    | none => .error (.sourceReject "temporal prediction basis shape")
  affineScore model.state (c * rank) (ERPFeatures.temporalFeatures matrix x)

theorem loaded_temporal_uses_actual_projection (p : TemporalPrimitives) (reference : ReferencePrimitives)
    (basis : FixedTemporalInput) (nfilter c : ℕ) (coefficient : Fin (c * basis.rank) → ℝ)
    (intercept : ℝ) (matrix : Matrix (Fin basis.times) (Fin basis.rank) ℝ)
    (x : ERPFeatures.Epoch c basis.times) (h : FixedTemporalConditions p basis)
    (hn : 0 < nfilter) (hc : 0 < c) (hmatrix : basis.matrix = matrixTensor matrix) :
    (fromState p reference (temporalERPRecord p basis nfilter c)
      (temporalNumericState coefficient intercept basis.matrix)).bind
      (fun model => temporalAffineScore c basis.times basis.rank model x) =
      .ok (ERPFeatures.evaluate ⟨coefficient, intercept⟩ (ERPFeatures.temporalFeatures matrix x)) := by
  have hconstruct := temporal_family_constructs p reference basis nfilter c coefficient intercept
    (List.ofFn (fun flat : Fin (basis.times * basis.rank) =>
      let pair := finProdFinEquiv.symm flat
      matrix pair.1 pair.2)) h hn hc (by simp) hmatrix
  rw [hconstruct]
  simp [temporalAffineScore, fixedTemporalModel, hmatrix, matrix_field_reads_all_entries,
    temporalNumericState, affine_score_with_declared_tail]

theorem bad_schema_precedes_temporal_read (p : TemporalPrimitives) (ref : ReferencePrimitives) (state : NamedState) :
    fromState p ref (Fields.ofList [("schema", .text "wrong"), ("feature_order", .text "channel_prime"),
      ("temporal_basis", .null)]) state = .error (.sourceReject "ERP schema/order") := by
  simp [fromState, fieldDefault, Fields.ofList, Fields.get, ensure]

def badScalarState : NamedState :=
  [("coefficient", coefficientTensor fixtureCoefficient), ("intercept", realTensor [1] [-1])]
def badWidthState : NamedState :=
  [("coefficient", realTensor [2] [2, 2]), ("intercept", interceptTensor (-1))]

theorem rank_one_intercept_full_reader_rejects :
    fromState fixturePrimitives fixtureReference (plainRecord .pooled 1 1 1 1 (.text "none")) badScalarState =
      .error (.sourceReject "ERP state shape") := by
  simp [fromState, plainRecord, fieldDefault, Fields.ofList, Fields.get, optionalTemporal,
    commonHeader, readFamily, Family.text, readGeometry, positiveInt, required, ensure,
    exactStateKeys, expectedKeys, stateKeys, badScalarState, normalizedState, normalizedTensor,
    coefficientTensor, realTensor,
    normalizedValues, numericScalar, tensorAt, lookupTensor, shapeAt, featureWidth,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, encodeNats]

theorem incorrect_coefficient_width_full_reader_rejects :
    fromState fixturePrimitives fixtureReference (plainRecord .pooled 1 1 1 1 (.text "none")) badWidthState =
      .error (.sourceReject "ERP state shape") := by
  simp [fromState, plainRecord, fieldDefault, Fields.ofList, Fields.get, optionalTemporal,
    commonHeader, readFamily, Family.text, readGeometry, positiveInt, required, ensure,
    exactStateKeys, expectedKeys, stateKeys, badWidthState, normalizedState, normalizedTensor,
    realTensor, interceptTensor, normalizedValues, numericScalar, tensorAt, lookupTensor,
    shapeAt, featureWidth, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, encodeNats]

def extraERPState : NamedState := fixtureLinearState ++ [("unrecognized", realTensor [1] [9])]

theorem extra_erp_key_full_reader_rejects :
    fromState fixturePrimitives fixtureReference (plainRecord .pooled 1 1 1 1 (.text "none")) extraERPState =
      .error (.sourceReject "ERP state keys") := by
  simp [fromState, plainRecord, fieldDefault, Fields.ofList, Fields.get, optionalTemporal,
    commonHeader, readFamily, Family.text, readGeometry, positiveInt, required, ensure,
    exactStateKeys, expectedKeys, stateKeys, extraERPState, fixtureLinearState, linearState,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, encodeNats]

theorem toeplitz_baseline_is_enforced :
    fromState fixturePrimitives fixtureReference (plainRecord .toeplitz 1 1 1 1 (.text "mean_only"))
      fixtureLinearState = .error (.sourceReject "Toeplitz baseline") := by
  simp [fromState, plainRecord, fieldDefault, Fields.ofList, Fields.get, optionalTemporal,
    commonHeader, readFamily, Family.text, positiveInt, required, ensure]

theorem invalid_xdawn_effective_count_precedes_reference :
    fromState fixturePrimitives fixtureReference (xdawnRecord 1 1 1 1 1 2 (.text "none")) fixtureXDawnState =
      .error (.sourceReject "xDAWN rank/effective") := by
  simp [fromState, xdawnRecord, plainRecord, fieldDefault, Fields.ofList, Fields.get, optionalTemporal,
    commonHeader, readFamily, Family.text, readGeometry, positiveInt, required, ensure,
    exactStateKeys, expectedKeys, stateKeys, fixtureXDawnState, xdawnNumericState,
    normalizedState, normalizedTensor, realTensor, coefficientTensor, interceptTensor,
    normalizedValues, numericScalar,
    PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, encodeNats]

def changedBasisState : NamedState := temporalNumericState fixtureCoefficient (-1)
  (realTensor [1, 1] [1 + 1 / 1000000000000])

theorem inconsistent_duplicate_temporal_basis_rejected :
    fromState fixturePrimitives fixtureReference (temporalERPRecord fixturePrimitives fixtureFixedInput 1 1)
      changedBasisState = .error (.sourceReject "temporal duplicate basis equality") := by
  have hreader := fixed_temporal_reader_from_explicit_fields fixturePrimitives fixtureFixedInput fixture_fixed_conditions
  have hh := temporal_common_header fixturePrimitives fixtureFixedInput 1 1 fixture_fixed_conditions (by decide)
  have hs : fieldDefault (temporalERPRecord fixturePrimitives fixtureFixedInput 1 1) "schema" =
      .text "n2p3_structured_erp/1" := rfl
  have ho : fieldDefault (temporalERPRecord fixturePrimitives fixtureFixedInput 1 1) "feature_order" =
      .text "channel_prime" := rfl
  have hopt : optionalTemporal fixturePrimitives (temporalERPRecord fixturePrimitives fixtureFixedInput 1 1) =
      .ok (some (fixedTemporalModel fixturePrimitives fixtureFixedInput)) := by
    simp [optionalTemporal, temporalERPRecord, Fields.get, hreader]
  simp only [fromState, hs, ho]
  simp only [ensure, hopt, hh, Bind.bind, Except.bind, Pure.pure, Except.pure]
  simp [temporalERPRecord, plainRecord, fieldDefault, Fields.ofList, Fields.get,
    readGeometry, required, ensure,
    exactStateKeys, expectedKeys, stateKeys, changedBasisState, temporalNumericState, linearState,
    normalizedState, normalizedTensor, realTensor, coefficientTensor, interceptTensor,
    normalizedValues, numericScalar,
    fixedTemporalModel, fixtureFixedInput, fixtureBasis, tensorAt, lookupTensor,
    sameNumericArray, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue, encodeNats]

theorem missing_nfilter_precedes_constructor_validation (p : TemporalPrimitives) (record : Fields)
    (temporal : Option TemporalModel) (family width baseline : Value)
    (hf : record.get "family" = some family) (hw : record.get "bin_width" = some width)
    (hb : record.get "baseline_mode" = some baseline) (hn : record.get "nfilter" = none) :
    commonHeader p record temporal = .error (.fieldFailure "missing:nfilter") := by
  simp [commonHeader, required, hf, hw, hb, hn]

theorem missing_nfilter_precedes_bad_family_in_full_reader (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (state : NamedState) :
    fromState p ref (Fields.ofList [("schema", .text "n2p3_structured_erp/1"),
      ("feature_order", .text "channel_prime"), ("family", .text "not-a-family"),
      ("bin_width", .boolean false), ("baseline_mode", .text "none")]) state =
      .error (.fieldFailure "missing:nfilter") := by
  simp [fromState, optionalTemporal, commonHeader, fieldDefault, Fields.ofList, Fields.get, required, ensure]

def canonicalTemporalERP (p : TemporalPrimitives) (basis : FixedTemporalInput) (nfilter c : ℕ)
    (coefficient : Fin (c * basis.rank) → ℝ) (intercept : ℝ) : Model :=
  ⟨⟨.temporal, 1, nfilter, basis.baseline, some (fixedTemporalModel p basis)⟩,
    c, basis.times, temporalNumericState coefficient intercept basis.matrix, none, none⟩

theorem temporal_export_record_fields (p : TemporalPrimitives) (basis : FixedTemporalInput) (nfilter c : ℕ)
    (coefficient : Fin (c * basis.rank) → ℝ) (intercept : ℝ) (key : String) :
    (exportRecord (canonicalTemporalERP p basis nfilter c coefficient intercept)).get key =
      (temporalERPRecord p basis nfilter c).get key := by
  by_cases hk : key = "temporal_basis"
  · subst key
    simp [exportRecord, canonicalTemporalERP, temporalERPRecord, plainRecord, Fields.ofList, Fields.get,
      fixedTemporalModel]
  · simp [exportRecord, canonicalTemporalERP, temporalERPRecord, plainRecord, Fields.ofList, Fields.get,
      fixedTemporalModel, hk]

theorem temporal_export_load_replay (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (basis : FixedTemporalInput) (nfilter c : ℕ) (coefficient : Fin (c * basis.rank) → ℝ)
    (intercept : ℝ) (values : List ℝ) (h : FixedTemporalConditions p basis)
    (hn : 0 < nfilter) (hc : 0 < c) (hlen : values.length = basis.times * basis.rank)
    (hmatrix : basis.matrix = realTensor [basis.times, basis.rank] values) :
    let model := canonicalTemporalERP p basis nfilter c coefficient intercept
    fromState p ref (exportState model).1 (exportState model).2 = .ok model := by
  dsimp only
  rw [exported_numeric_fields_are_preserved]
  dsimp only [exportState]
  rw [fromState_record_lookup_ext p ref _ (temporalERPRecord p basis nfilter c) _
    (temporal_export_record_fields p basis nfilter c coefficient intercept)]
  exact temporal_family_constructs p ref basis nfilter c coefficient intercept values h hn hc hlen hmatrix

def canonicalXDawnERP (width nfilter c t rank effective : ℕ) (baseline : Value)
    (filters prototypes reference : List ℝ)
    (coefficient : Fin ((4 * effective) * (4 * effective + 1) / 2) → ℝ) (intercept : ℝ) : Model :=
  ⟨⟨.xdawn, width, nfilter, baseline, none⟩, c, t,
    xdawnNumericState c (t / width) effective filters prototypes reference coefficient intercept,
    some rank, some effective⟩

theorem xdawn_export_record_fields (width nfilter c t rank effective : ℕ) (baseline : Value)
    (filters prototypes reference : List ℝ)
    (coefficient : Fin ((4 * effective) * (4 * effective + 1) / 2) → ℝ) (intercept : ℝ) (key : String) :
    (exportRecord (canonicalXDawnERP width nfilter c t rank effective baseline
      filters prototypes reference coefficient intercept)).get key =
      (xdawnRecord width nfilter c t rank effective baseline).get key := by
  by_cases hr : key = "observed_rank"
  · subst key
    simp [exportRecord, canonicalXDawnERP, xdawnRecord, plainRecord, Fields.ofList, Fields.get]
  · by_cases he : key = "effective_nfilter"
    · subst key
      simp [exportRecord, canonicalXDawnERP, xdawnRecord, plainRecord, Fields.ofList, Fields.get]
    · simp [exportRecord, canonicalXDawnERP, xdawnRecord, plainRecord, Fields.ofList, Fields.get, hr, he]

theorem xdawn_export_load_replay (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (width nfilter c t rank effective : ℕ) (baseline : Value) (filters prototypes reference : List ℝ)
    (coefficient : Fin ((4 * effective) * (4 * effective + 1) / 2) → ℝ) (intercept : ℝ)
    (hw : 0 < width) (hn : 0 < nfilter) (hc : 0 < c) (ht : 0 < t) (hb : 0 < t / width)
    (hr : 0 < rank ∧ rank ≤ c) (he : 0 < effective ∧ effective = min nfilter rank)
    (hfLen : filters.length = (2 * effective) * c)
    (hpLen : prototypes.length = (2 * effective) * (t / width))
    (hrLen : reference.length = (4 * effective) * (4 * effective))
    (hclose : ref.symmetryAllclose (realTensor [4 * effective, 4 * effective] reference) = true)
    (hpositive : ref.eigenvaluesPositive (realTensor [4 * effective, 4 * effective] reference) = true) :
    let model := canonicalXDawnERP width nfilter c t rank effective baseline
      filters prototypes reference coefficient intercept
    fromState p ref (exportState model).1 (exportState model).2 = .ok model := by
  dsimp only
  rw [exported_numeric_fields_are_preserved]
  dsimp only [exportState]
  rw [fromState_record_lookup_ext p ref _ (xdawnRecord width nfilter c t rank effective baseline) _
    (xdawn_export_record_fields width nfilter c t rank effective baseline filters prototypes reference coefficient intercept)]
  exact xdawn_family_constructs p ref width nfilter c t rank effective baseline filters prototypes reference
    coefficient intercept hw hn hc ht hb hr he hfLen hpLen hrLen hclose hpositive

/-- `nonfinite` is an output of a numerical primitive outside the finite-real
model, not an invented source refusal. The actual zero-augmentation path can
return NaN without an exception. Exceptions remain distinct primitive results. -/
inductive NumericOutcome (α : Type*) where
  | finite (value : α)
  | nonfinite
  | exception (operation : String)

def numericThen {α β : Type*} (value : NumericOutcome α) (next : α → NumericOutcome β) : NumericOutcome β :=
  match value with
  | .finite value => next value
  | .nonfinite => .nonfinite
  | .exception operation => .exception operation

def numericMap {α β : Type*} (f : α → β) (value : NumericOutcome α) : NumericOutcome β :=
  numericThen value (fun a => .finite (f a))

structure XDawnNumerics where
  oasCovariance : (dimension times : ℕ) → ERPFeatures.Epoch dimension times →
    NumericOutcome (Matrix (Fin dimension) (Fin dimension) ℝ)
  tangent : (dimension : ℕ) → Matrix (Fin dimension) (Fin dimension) ℝ →
    Matrix (Fin dimension) (Fin dimension) ℝ →
    NumericOutcome (Fin (dimension * (dimension + 1) / 2) → ℝ)

def spatialTransform {c v bins : ℕ} (filters : Matrix (Fin v) (Fin c) ℝ)
    (epoch : ERPFeatures.Epoch c bins) : ERPFeatures.Epoch v bins :=
  fun i t => ∑ channel, filters i channel * epoch channel t

def prototypeFirst {effective bins : ℕ} (prototypes transformed : ERPFeatures.Epoch (2 * effective) bins) :
    ERPFeatures.Epoch (4 * effective) bins := fun i t =>
  if hi : i.val < 2 * effective then prototypes ⟨i.val, hi⟩ t
  else transformed ⟨i.val - 2 * effective, by omega⟩ t

def xdawnSteps {c bins effective : ℕ} (p : XDawnNumerics)
    (filters : Matrix (Fin (2 * effective)) (Fin c) ℝ)
    (prototypes : ERPFeatures.Epoch (2 * effective) bins)
    (reference : Matrix (Fin (4 * effective)) (Fin (4 * effective)) ℝ)
    (affine : ERPFeatures.Affine ((4 * effective) * (4 * effective + 1) / 2))
    (epoch : ERPFeatures.Epoch c bins) : NumericOutcome ℝ :=
  numericThen (p.oasCovariance (4 * effective) bins (prototypeFirst prototypes (spatialTransform filters epoch)))
    (fun covariance => numericMap (ERPFeatures.evaluate affine)
      (p.tangent (4 * effective) reference covariance))

def matrixAt (rows cols : ℕ) (state : NamedState) (name : String) :
    Check (Matrix (Fin rows) (Fin cols) ℝ) := do
  let tensor ← tensorAt state name
  match matrixField rows cols tensor with
  | some matrix => .ok matrix
  | none => .error (.fieldFailure ("finite ERP matrix decoding:" ++ name))

def affineAt (d : ℕ) (state : NamedState) : Check (ERPFeatures.Affine d) := do
  let tensor ← tensorAt state "coefficient"
  let coefficient ← match PersonalRestore.tensorVector d tensor with
    | some coefficient => .ok coefficient
    | none => .error (.fieldFailure "finite ERP coefficient decoding")
  let tensor ← tensorAt state "intercept"
  let intercept ← match strictScalar tensor with
    | some intercept => .ok intercept
    | none => .error (.fieldFailure "finite ERP scalar decoding")
  pure ⟨coefficient, intercept⟩

/-- This extracts the individually guarded finite fields before following
source spatial multiplication, prototype-first augmentation, OAS and tangent.
It does not add a positive-definite or finite-future gate to Python prediction. -/
def xdawnForward (c bins effective : ℕ) (p : XDawnNumerics) (state : NamedState)
    (epoch : ERPFeatures.Epoch c bins) : Check (NumericOutcome ℝ) := do
  let filters ← matrixAt (2 * effective) c state "filters"
  let prototypes ← matrixAt (2 * effective) bins state "prototypes"
  let reference ← matrixAt (4 * effective) (4 * effective) state "reference"
  let affine ← affineAt ((4 * effective) * (4 * effective + 1) / 2) state
  pure (xdawnSteps p filters prototypes reference affine epoch)

def xdawnMatrixState {c bins effective : ℕ}
    (filters : Matrix (Fin (2 * effective)) (Fin c) ℝ)
    (prototypes : ERPFeatures.Epoch (2 * effective) bins)
    (reference : Matrix (Fin (4 * effective)) (Fin (4 * effective)) ℝ)
    (coefficient : Fin ((4 * effective) * (4 * effective + 1) / 2) → ℝ) (intercept : ℝ) : NamedState :=
  linearState coefficient intercept ++ [("filters", matrixTensor filters),
    ("prototypes", matrixTensor prototypes), ("reference", matrixTensor reference)]

theorem affine_fields_recovered_with_tail {d : ℕ} (coefficient : Fin d → ℝ) (intercept : ℝ) (tail : NamedState) :
    affineAt d (linearState coefficient intercept ++ tail) = .ok ⟨coefficient, intercept⟩ := by
  simp [affineAt, tensorAt, linearState, lookupTensor, PersonalRestore.tensorVector,
    coefficientTensor, interceptTensor, realTensor, strictScalar, numericScalar,
    List.map_ofFn, Function.comp_def, PersonalRestore.decodeVector_ofFn]

theorem xdawn_forward_from_individually_decoded_fields {c bins effective : ℕ}
    (p : XDawnNumerics) (filters : Matrix (Fin (2 * effective)) (Fin c) ℝ)
    (prototypes : ERPFeatures.Epoch (2 * effective) bins)
    (reference : Matrix (Fin (4 * effective)) (Fin (4 * effective)) ℝ)
    (coefficient : Fin ((4 * effective) * (4 * effective + 1) / 2) → ℝ) (intercept : ℝ)
    (epoch : ERPFeatures.Epoch c bins) :
    xdawnForward c bins effective p (xdawnMatrixState filters prototypes reference coefficient intercept) epoch =
      .ok (xdawnSteps p filters prototypes reference ⟨coefficient, intercept⟩ epoch) := by
  simp [xdawnForward, xdawnMatrixState, matrixAt, tensorAt, linearState, lookupTensor,
    matrix_field_reads_all_entries, affineAt, PersonalRestore.tensorVector, coefficientTensor,
    interceptTensor, realTensor, strictScalar, numericScalar, List.map_ofFn, Function.comp_def,
    PersonalRestore.decodeVector_ofFn]

theorem loaded_xdawn_reaches_actual_spatial_augmentation_steps (p : TemporalPrimitives) (ref : ReferencePrimitives)
    (numeric : XDawnNumerics) (width nfilter c t rank effective : ℕ) (baseline : Value)
    (filters : Matrix (Fin (2 * effective)) (Fin c) ℝ)
    (prototypes : ERPFeatures.Epoch (2 * effective) (t / width))
    (reference : Matrix (Fin (4 * effective)) (Fin (4 * effective)) ℝ)
    (coefficient : Fin ((4 * effective) * (4 * effective + 1) / 2) → ℝ) (intercept : ℝ)
    (epoch : ERPFeatures.Epoch c t)
    (hw : 0 < width) (hn : 0 < nfilter) (hc : 0 < c) (ht : 0 < t) (hb : 0 < t / width)
    (hr : 0 < rank ∧ rank ≤ c) (he : 0 < effective ∧ effective = min nfilter rank)
    (hclose : ref.symmetryAllclose (matrixTensor reference) = true)
    (hpositive : ref.eigenvaluesPositive (matrixTensor reference) = true) :
    (fromState p ref (xdawnRecord width nfilter c t rank effective baseline)
      (xdawnMatrixState filters prototypes reference coefficient intercept)).bind
      (fun model => xdawnForward c (t / width) effective numeric model.state (ERPFeatures.completeBins width epoch)) =
      .ok (xdawnSteps numeric filters prototypes reference ⟨coefficient, intercept⟩ (ERPFeatures.completeBins width epoch)) := by
  have hconstruct := xdawn_family_constructs p ref width nfilter c t rank effective baseline
    (List.ofFn (fun flat : Fin ((2 * effective) * c) => let pair := finProdFinEquiv.symm flat; filters pair.1 pair.2))
    (List.ofFn (fun flat : Fin ((2 * effective) * (t / width)) => let pair := finProdFinEquiv.symm flat; prototypes pair.1 pair.2))
    (List.ofFn (fun flat : Fin ((4 * effective) * (4 * effective)) => let pair := finProdFinEquiv.symm flat; reference pair.1 pair.2))
    coefficient intercept hw hn hc ht hb hr he (by simp) (by simp) (by simp) hclose hpositive
  change (fromState p ref _ (xdawnNumericState c (t / width) effective _ _ _ coefficient intercept)).bind _ = _
  rw [hconstruct]
  exact xdawn_forward_from_individually_decoded_fields numeric filters prototypes reference coefficient intercept _

end

end N2P3.ERPState
