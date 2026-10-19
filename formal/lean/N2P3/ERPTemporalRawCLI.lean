import N2P3.ERPTemporalAttestedLoad

/-!
# Raw fit_plan preparation before any full Dataset validation

Source: fit_source_temporal_basis.py:27-49. JSON values and raw array shapes
are retained through the literal path/load/row/index/identity sequence. The
attested loader here is its explicit program, not a load-success primitive.
Neither a full Dataset check nor an N/C/T Context is constructed at this stage.

In particular X.shape[0] is read while constructing the "all" range, or only
after the short-circuit rank/kind/nonempty/negative-row tests for other rows.
subject_ids[rows] is one native indexing operation and can fail even when rows
were valid against X.shape[0]. unique, astype(str), tolist and identity subset
then follow. Path, codec and individual ndarray implementations remain local
unrefined boundaries; no leaf produces a fitted source or an admitted scope.
-/

namespace N2P3.ERPTemporalRawCLI

open N2P3.FrozenState N2P3.StateLedger
open N2P3.ERPTemporalCache N2P3.ERPTemporalSnapshot N2P3.ERPTemporalNPZ
open N2P3.ERPTemporalJSON N2P3.ERPTemporalArrayViews

noncomputable section

attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Operations (World : Type) where
  load : ERPTemporalAttestedLoad.Operations World
  path : JValue → EStateM Fault World String
  readUTF8 : String → EStateM Fault World String
  pathExists : String → EStateM Fault World Bool
  isAbsolute : String → Bool
  resolve : String → EStateM Fault World String
  parent : String → String
  join : String → String → String
  asRows : JValue → EStateM Fault World RawArray
  arange : ℕ → EStateM Fault World RawArray
  indexSubjects : (original : RawArray) → RawArray → EStateM Fault World
    {image : RawArray // image.kind=original.kind}

structure Header where
  raw : JFields
  excluded : List String

def excludedStrings : List JValue → Check (List String)
  | [] => .ok []
  | .text value :: tail => do
    ensure (value ≠ "") "source temporal excluded study list"
    let rest ← excludedStrings tail
    pure (value :: rest)
  | _ :: _ => .error (.sourceReject "source temporal excluded study list")

def checkHeader (raw : JValue) : Check Header := do
  let fields ← ERPTemporalRawFields.objectAt raw "source temporal plan.get"
  ensure (pythonEqual (ERPTemporalRawFields.fieldDefault fields "schema")
    (.text "n2p3_source_temporal_plan/1")) "source temporal plan schema"
  let excluded ← match (← ERPTemporalRawFields.required fields "excluded_target_studies") with
    | .sequence values => .ok values.toList
    | _ => .error (.sourceReject "source temporal excluded study list")
  ensure (excluded ≠ []) "source temporal excluded study list"
  let excluded ← excludedStrings excluded
  pure ⟨fields,excluded⟩

/-- No list or nonempty requirement is added for plan["sources"]. -/
def sourceItems : JValue → Check (List JValue)
  | .sequence values => .ok values.toList
  | .object fields => .ok (fields.keys.map JValue.text)
  | .text value => .ok (value.toList.map (fun char => .text (String.singleton char)))
  | _ => .error (.fieldFailure "iterate source temporal sources")

structure Plan where
  path : String
  output : String
  header : Header
  items : List JValue

def preparePlan {World : Type} (p : Operations World) (planPath output : JValue) :
    TraceM World Plan := do
  let planPath ← small (p.path planPath)
  let output ← small (p.path output)
  let text ← small (p.readUTF8 planPath)
  let raw ← small (p.load.record.decode.decodeJSON text)
  let header ← checked (checkHeader raw)
  let occupied ← small (p.pathExists output)
  checked (ensure (occupied=false) "source temporal output already exists")
  let items ← checked (sourceItems (← checked (ERPTemporalRawFields.required header.raw "sources")))
  pure ⟨planPath,output,header,items⟩

def cachePath {World : Type} (p : Operations World) (planPath : String) (item : JValue) :
    TraceM World String := do
  let fields ← checked (ERPTemporalRawFields.objectAt item "source item cache subscription")
  let path ← small (p.path (← checked (ERPTemporalRawFields.required fields "cache")))
  if p.isAbsolute path then pure path
  else
    let resolved ← small (p.resolve planPath)
    pure (p.join (p.parent resolved) path)

def epochCount (dataset : ERPTemporalRawCacheDecode.Dataset) : Check ℕ :=
  match dataset.X.1.shape with
  | [] => .error (.fieldFailure "dataset.n_epochs X.shape[0]")
  | first :: _ => .ok first

def integerCell : Cell → ℤ
  | .atom (.integer value) => value
  | _ => 0

def rowIntegers (rows : RawArray) : List ℤ := rows.cells.map integerCell

/-- integerCell's fallback is never a cast/admission rule: the dtype-kind
gate executes first. Original raw array is retained for producer re-coercion. -/
def checkRowPrefix (rows : RawArray) : Check Unit := do
  ensure (rows.shape.length=1) "source temporal explicit valid rows"
  ensure (rows.kind.integer=true) "source temporal explicit valid rows"
  let length ← arrayLength rows
  ensure (length ≠ 0) "source temporal explicit valid rows"
  ensure (∀ row ∈ rowIntegers rows, 0 ≤ row) "source temporal explicit valid rows"

def checkRows {World : Type} (dataset : ERPTemporalRawCacheDecode.Dataset)
    (rows : RawArray) : TraceM World Unit := do
  checked (checkRowPrefix rows)
  let epochs ← checked (epochCount dataset)
  checked (ensure (∀ row ∈ rowIntegers rows, row < (epochs : ℤ)) "source temporal explicit valid rows")

def prepareRows {World : Type} (p : Operations World)
    (dataset : ERPTemporalRawCacheDecode.Dataset) (item : JValue) : TraceM World RawArray := do
  let fields ← checked (ERPTemporalRawFields.objectAt item "source item rows subscription")
  let raw ← checked (ERPTemporalRawFields.required fields "rows")
  let rows ← if pythonEqual raw (.text "all") then do
      small (p.arange (← checked (epochCount dataset)))
    else small (p.asRows raw)
  checkRows dataset rows
  pure rows

def selectedStudies {World : Type} (p : Operations World)
    (dataset : ERPTemporalRawCacheDecode.Dataset) (rows : RawArray) : TraceM World (List String) := do
  let gathered ← small (p.indexSubjects dataset.subjects.1 rows)
  -- np.unique is applied before astype(str), and flattens the gathered array.
  let unique ← small (p.load.record.uniqueStrings ⟨gathered.1,gathered.2.trans dataset.subjects.2⟩)
  let converted ← small (p.load.record.astype .string unique.1)
  -- unique's rank-one result and astype's preserved shape make these values
  -- exactly the .tolist() string list, rather than a flatten of raw subjects.
  let requested := (stringArray ⟨converted.1,converted.2.1⟩).values
  let selected ← checked (ERPTemporalCLI.subsetIdentity p.load.record.decode.input dataset.identity requested)
  let studies ← checked (PersonalERPInput.allIdentityStudies p.load.record.decode.input selected)
  pure (studies.dedup.mergeSort (· ≤ ·))

structure Scope where
  dataset : ERPTemporalRawCacheDecode.Dataset
  rows : RawArray

def prepareScope {World : Type} (p : Operations World) (planPath : String)
    (excluded : List String) (item : JValue) : TraceM World Scope := do
  let path ← cachePath p planPath item
  let dataset ← ERPTemporalAttestedLoad.load p.load path true
  let rows ← prepareRows p dataset item
  let studies ← selectedStudies p dataset rows
  checked (ERPTemporalCLI.checkSelectedExclusion excluded studies)
  pure ⟨dataset,rows⟩

def prepareScopes {World : Type} (p : Operations World) (planPath : String)
    (excluded : List String) : List JValue → TraceM World (List Scope)
  | [] => pure []
  | item :: tail => do
    let scope ← prepareScope p planPath excluded item
    let rest ← prepareScopes p planPath excluded tail
    pure (scope :: rest)

structure Preparation where
  plan : Plan
  scopes : List Scope

/-- rank is intentionally not subscribed yet. The following actual source
fit is called even when this completes with no scopes. -/
def prepare {World : Type} (p : Operations World) (planPath output : JValue) :
    TraceM World Preparation := do
  let plan ← preparePlan p planPath output
  let scopes ← prepareScopes p plan.path plan.header.excluded plan.items
  pure ⟨plan,scopes⟩

end
end N2P3.ERPTemporalRawCLI
