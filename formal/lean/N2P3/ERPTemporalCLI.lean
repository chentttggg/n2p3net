import N2P3.ERPTemporalProducer
import N2P3.PersonalERPInput
import Mathlib.Data.List.FinRange
import Lean.Elab.Tactic.Omega

/-!
# Literal source-temporal plan frontend and requested rows

Source: experiments/fit_source_temporal_basis.py:27-55. This module starts
with independently decoded finite, unique-key JSON fields and expands the
actual schema, excluded strings, output-exists, sources iteration, path,
row-array and selected-identity operations. It does not assert that the
attested loader runs the full EpochDataset validator: that check is performed
later by each fit_source iteration. The remaining snapshot/NPZ/IO and full
producer composition are separate obligations, not a validated-load callback.

Path/UTF8/JSON operations and NumPy's one-array coercion/index operation are
small boundaries with their own failures. A row image records dtype kind,
dimension and the integer coordinates used only after an integer-kind gate.
Its noninteger cell values are immaterial because that kind is refused first.
This does not implement arbitrary NumPy coercions or IEEE behavior.

In particular, sources is ITERATED, not required to be a nonempty list;
excluded strings are neither stripped nor required unique; CLI rows need not
be unique; and the raw rank field is retained without being read here. These
weak and delayed gates agree with the actual source, not a desired contract.
-/

namespace N2P3.ERPTemporalCLI

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPInput

noncomputable section

attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Header where
  fields : Fields
  excluded : List String

def excludedStrings : List Value → Check (List String)
  | [] => .ok []
  | value :: tail => do
    let text ← match value with
      | .text s => .ok s
      | _ => .error (.sourceReject "source temporal excluded study list")
    ensure (text ≠ "") "source temporal excluded study list"
    let rest ← excludedStrings tail
    pure (text :: rest)

def checkHeader (raw : Value) : Check Header := do
  let fields ← objectAt raw "source temporal plan.get"
  ensure (fieldDefault fields "schema" = .text "n2p3_source_temporal_plan/1") "source temporal plan schema"
  let excluded ← match (← required fields "excluded_target_studies") with
    | .sequence values => .ok (valuesToList values)
    | _ => .error (.sourceReject "source temporal excluded study list")
  ensure (excluded ≠ []) "source temporal excluded study list"
  let excluded ← excludedStrings excluded
  pure ⟨fields, excluded⟩

/-- Python iteration on already-decoded JSON objects. Empty object/string
sources are empty iterables; scalar noniterability is a field operation error.
Object key order is retained. No sources-list validation is invented. -/
def sourceItems : Value → Check (List Value)
  | .sequence values => .ok (valuesToList values)
  | .object fields => .ok ((fieldKeys fields).map Value.text)
  | .text s => .ok (s.toList.map (fun c => .text (String.singleton c)))
  | _ => .error (.fieldFailure "iterate source temporal sources")

structure PathPrimitives where
  path : Value → Check String
  isAbsolute : String → Bool
  resolveParent : String → Check String
  join : String → String → String
  pathExists : String → Check Bool
  readUTF8 : String → Check String
  decodeJSON : String → Check Value

def prepareAfterHeader (p : PathPrimitives) (output : String) (header : Header) :
    Check (List Value) := do
  let occupied ← p.pathExists output
  ensure (occupied = false) "source temporal output already exists"
  sourceItems (← required header.fields "sources")

def preparePlan (p : PathPrimitives) (planPath output : Value) : Check (Header × List Value) := do
  let planPath ← p.path planPath
  let output ← p.path output
  let text ← p.readUTF8 planPath
  let raw ← p.decodeJSON text
  let header ← checkHeader raw
  let items ← prepareAfterHeader p output header
  pure (header,items)

def cachePath (p : PathPrimitives) (planPath : String) (item : Value) : Check String := do
  let fields ← objectAt item "source item cache subscription"
  let cache ← p.path (← required fields "cache")
  if p.isAbsolute cache then pure cache
  else
    let parent ← p.resolveParent planPath
    pure (p.join parent cache)

inductive DTypeKind where
  | signed | unsigned | boolean | floating | other
  deriving DecidableEq

structure RowArray where
  ndim : ℕ
  kind : DTypeKind
  integers : List ℤ

/-- NumPy np.asarray on exactly one requested-row object, not a complete
plan/cache/source-admission operation. The actual ndarray domain is separate
from raw JSON; the producer will re-coerce the retained array image. -/
inductive RowObject where
  | json (raw : Value)
  | array (image : RowArray)

abbrev ArrayCoercion := RowObject → Check RowArray

def rowArray (coerce : ArrayCoercion) (epochs : ℕ) (raw : Value) : Check RowArray :=
  match raw with
  | .text "all" => .ok ⟨1,.signed,(List.range epochs).map Int.ofNat⟩
  | _ => coerce (.json raw)

def InRange (epochs : ℕ) (rows : List ℤ) : Prop :=
  ∀ row ∈ rows, 0 ≤ row ∧ row < (epochs : ℤ)

def boundedRows (epochs : ℕ) : (rows : List ℤ) → InRange epochs rows → List (Fin epochs)
  | [], _ => []
  | row :: tail, checked =>
    ⟨row.toNat, by have := checked row (by simp); omega⟩ ::
      boundedRows epochs tail (fun value mem => checked value (by simp [mem]))

def BasicRows (epochs : ℕ) (image : RowArray) : Prop :=
  image.ndim = 1 ∧ (image.kind = .signed ∨ image.kind = .unsigned) ∧
    image.integers ≠ [] ∧ InRange epochs image.integers

structure SelectedRows (epochs : ℕ) where
  image : RowArray
  indices : List (Fin epochs)

def checkCLIRows (epochs : ℕ) (image : RowArray) : Check (SelectedRows epochs) := by
  classical
  exact if checked : BasicRows epochs image then
    .ok ⟨image, boundedRows epochs image.integers checked.2.2.2⟩
  else .error (.sourceReject "source temporal explicit valid rows")

def checkProducerRows (epochs : ℕ) (image : RowArray) : Check (SelectedRows epochs) := by
  classical
  exact if checked : BasicRows epochs image ∧ image.integers.Nodup then
    .ok ⟨image, boundedRows epochs image.integers checked.1.2.2.2⟩
  else .error (.sourceReject "source temporal explicit unique in-range rows")

def prepareRows (coerce : ArrayCoercion) (epochs : ℕ) (item : Value) : Check (SelectedRows epochs) := do
  let fields ← objectAt item "source item rows subscription"
  let image ← rowArray coerce epochs (← required fields "rows")
  checkCLIRows epochs image

/-- This is the source table.subset construction, not materializeIdentity.
The attested CLI branch accesses the table already decoded from identity_json.
The general API's no-table materialization belongs to the later fit loop. -/
def subsetIdentity (p : PersonalERPInput.Primitives) (rows : List IdentityRecord)
    (requested : List String) : Check (List IdentityRecord) := do
  let requested ← cleanStrings p requested
  let requested := requested.dedup
  ensure (∀ subject ∈ requested, subject ∈ rows.map IdentityRecord.localSubject)
    "identity subset unknown local subject"
  ensure (requested ≠ []) "identity subset cannot be empty"
  checkIdentityTableConstructor (rows.filter (fun row => row.localSubject ∈ requested))

/-- Source string-array advanced indexing is one NumPy operation. Shape,
indexing exceptions and byte representation are not hidden inside a fit or
identity predicate. np.unique on the resulting string cells is explicit. -/
abbrev IndexSubjects := PersonalERPData.ArrayImage String → List ℤ →
  Check (PersonalERPData.ArrayImage String)

def selectedStudies (p : PersonalERPInput.Primitives) (indexSubjects : IndexSubjects)
    {N C T M : ℕ} (context : Context N C T M) (rows : SelectedRows N) : Check (List String) := do
  let gathered ← indexSubjects context.data.subjectImage rows.image.integers
  let subjects := gathered.values.dedup.mergeSort (· ≤ ·)
  let table ← match context.identityCache with
    | some table => .ok table
    | none => .error (.fieldFailure "loaded identity_table.subset")
  let selected ← subsetIdentity p table subjects
  let studies ← allIdentityStudies p selected
  pure (studies.dedup.mergeSort (· ≤ ·))

def checkSelectedExclusion (excluded studies : List String) : Check Unit :=
  ensure (∀ study ∈ studies, study ∉ excluded) "source temporal excluded target study"

theorem excluded_empty_string_refused : excludedStrings [.text ""] =
    .error (.sourceReject "source temporal excluded study list") := by
  simp [excludedStrings,ensure]

theorem excluded_whitespace_and_duplicates_are_not_newly_forbidden :
    excludedStrings [.text " ",.text " "] = .ok [" "," "] := by
  simp [excludedStrings,ensure]

theorem empty_object_sources_iterates_to_empty : sourceItems (.object .nil) = .ok [] := rfl

theorem scalar_sources_is_operation_failure : sourceItems (.integer 0) =
    .error (.fieldFailure "iterate source temporal sources") := rfl

theorem existing_output_precedes_sources_subscription (p : PathPrimitives)
    (header : Header) (output : String) (occupied : p.pathExists output = .ok true) :
    prepareAfterHeader p output header = .error (.sourceReject "source temporal output already exists") := by
  simp [prepareAfterHeader,occupied,ensure]

theorem absolute_cache_does_not_resolve_plan_parent (p : PathPrimitives)
    (planPath : String) (item : Fields) (cache : Value) (path : String)
    (field : item.get "cache" = some cache) (converted : p.path cache = .ok path)
    (absolute : p.isAbsolute path = true) :
    cachePath p planPath (.object item) = .ok path := by
  simp [cachePath,objectAt,required,field,converted,absolute]

theorem bounded_rows_preserve_integer_coordinates (epochs : ℕ) (rows : List ℤ)
    (checked : InRange epochs rows) :
    (boundedRows epochs rows checked).map (fun row => (row.val : ℤ)) = rows := by
  induction rows with
  | nil => rfl
  | cons row tail ih =>
    simp only [boundedRows,List.map_cons]
    rw [Int.toNat_of_nonneg (checked row (by simp)).1,ih]

theorem pure_boolean_row_array_refused (epochs : ℕ) (rows : List ℤ) :
    checkCLIRows epochs ⟨1,.boolean,rows⟩ =
      .error (.sourceReject "source temporal explicit valid rows") := by
  simp [checkCLIRows,BasicRows]

theorem float_row_array_is_not_silently_converted (epochs : ℕ) (rows : List ℤ) :
    checkCLIRows epochs ⟨1,.floating,rows⟩ =
      .error (.sourceReject "source temporal explicit valid rows") := by
  simp [checkCLIRows,BasicRows]

theorem CLI_duplicate_rows_pass_but_producer_refuses :
    (checkCLIRows 2 ⟨1,.signed,[0,0]⟩).map (fun selected => selected.indices.map Fin.val) = .ok [0,0] ∧
    checkProducerRows 2 ⟨1,.signed,[0,0]⟩ =
      .error (.sourceReject "source temporal explicit unique in-range rows") := by
  have hb : BasicRows 2 ⟨1,.signed,[0,0]⟩ := by simp [BasicRows,InRange]
  constructor
  · simp [checkCLIRows,hb,boundedRows]
  · simp [checkProducerRows,BasicRows,InRange]

theorem all_token_constructs_actual_complete_range (coerce : ArrayCoercion) (epochs : ℕ) :
    rowArray coerce epochs (.text "all") = .ok ⟨1,.signed,(List.range epochs).map Int.ofNat⟩ := by
  simp [rowArray]

theorem selected_study_overlap_refuses (excluded studies : List String) (study : String)
    (hs : study ∈ studies) (he : study ∈ excluded) :
    checkSelectedExclusion excluded studies = .error (.sourceReject "source temporal excluded target study") := by
  have hn : ¬∀ value ∈ studies, value ∉ excluded := by
    intro h
    exact h study hs he
  simp [checkSelectedExclusion,ensure,hn]

end
end N2P3.ERPTemporalCLI
