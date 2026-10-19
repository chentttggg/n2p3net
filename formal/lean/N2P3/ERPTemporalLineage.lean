import N2P3.ERPTemporalNPZ

/-! Literal finite-JSON DataLineage/LineageEntity reconstruction from
lineage.py:19-46,60-109,118-164. Parsing, canonical parent digests, each entity
digest, topological/duplicate checks, head and lineage digests are explicit.
This is not a constructor-success callback. Scalar Python str, strip/lower
and canonical JSON SHA256 are individual operations; nonfinite JSON and
arbitrary Python object/NumPy scalar methods are outside this Value domain. -/

namespace N2P3.ERPTemporalLineage

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData

noncomputable section

attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Primitives where
  pythonString : Value → Check String
  strip : String → String
  lower : String → String
  semanticHash : Value → Check String

mutual
def canonicalValue : Value → Check Value
  | .object fields => .object <$> canonicalFields fields
  | .sequence values => .sequence <$> canonicalValues values
  | value => .ok value
def canonicalFields : Fields → Check Fields
  | .nil => .ok .nil
  | .cons key value tail => do
    ensure (key ≠ "") "lineage parameter keys nonempty strings"
    let value ← canonicalValue value
    let tail ← canonicalFields tail
    pure (.cons key value tail)
def canonicalValues : Values → Check Values
  | .nil => .ok .nil
  | .cons value tail => do
    let value ← canonicalValue value
    let tail ← canonicalValues tail
    pure (.cons value tail)
end

def canonicalDigest (p : Primitives) (raw : Value) : Check String := do
  p.semanticHash (← canonicalValue raw)

def parentDigest (p : Primitives) (raw : String) : Check String := do
  let string ← p.pythonString (.text raw)
  let string := p.lower (p.strip string)
  ensure (LowerHex64 string) "lineage parent SHA256"
  pure string

def parentDigests (p : Primitives) : List String → Check (List String)
  | [] => .ok []
  | value :: tail => do
    let value ← parentDigest p value
    let tail ← parentDigests p tail
    pure (value :: tail)

structure Entity where
  operation : String
  parents : List String
  parameters : Fields

def entitySemantic (entity : Entity) : Check Fields := do
  let parameters ← canonicalFields entity.parameters
  pure (Fields.ofList [("operation",.text entity.operation),
    ("parent_entity_digests",.sequence (encodeTexts entity.parents)),("parameters",.object parameters)])

def entityDigest (p : Primitives) (entity : Entity) : Check String := do
  canonicalDigest p (.object (← entitySemantic entity))

def entityPayload (p : Primitives) (entity : Entity) : Check Fields := do
  let semantic ← entitySemantic entity
  let digest ← entityDigest p entity
  pure (.cons "entity_digest" (.text digest) semantic)

def constructEntity (p : Primitives) (operation : String) (parents : List String)
    (parameters : Fields) : Check Entity := do
  let operation := p.strip operation
  ensure (operation ≠ "") "lineage operation nonempty"
  let parents ← parentDigests p parents
  let parents := parents.dedup.mergeSort (· ≤ ·)
  ensure (operation="source_ingress" → parents=[]) "source lineage has no parents"
  ensure (operation ≠ "source_ingress" → parents ≠ []) "derived lineage needs parents"
  ensure (fieldKeys parameters ≠ []) "lineage parameters nonempty mapping"
  let _ ← canonicalFields parameters
  pure ⟨operation,parents,parameters⟩

def readEntity (p : Primitives) (raw : Value) : Check Entity := do
  let fields ← objectAt raw "lineage entity mapping"
  ensure (ExactKeys fields ["operation","parent_entity_digests","parameters","entity_digest"])
    "lineage entity exact fields"
  let parents ← PersonalERPExecution.textList (← required fields "parent_entity_digests") "lineage parents string list"
  let parameters ← objectAt (← required fields "parameters") "lineage entity parameters mapping"
  let operation ← p.pythonString (← required fields "operation")
  let entity ← constructEntity p operation parents parameters
  let digest ← entityDigest p entity
  ensure (fieldDefault fields "entity_digest"=.text digest) "lineage entity semantic digest"
  pure entity

def readEntities (p : Primitives) : List Value → Check (List Entity)
  | [] => .ok []
  | value :: tail => do
    let entity ← readEntity p value
    let rest ← readEntities p tail
    pure (entity :: rest)

def equalEntity (a b : Entity) : Prop :=
  a.operation=b.operation ∧ a.parents=b.parents ∧ PersonalAdmission.PyEq (.object a.parameters) (.object b.parameters)

def insertSeen (digest : String) (entity : Entity) (seen : List (String × Entity)) : List (String × Entity) :=
  if seen.any (fun item => item.1=digest) then
    seen.map (fun item => if item.1=digest then (digest,entity) else item)
  else seen ++ [(digest,entity)]

def checkEntities (p : Primitives) : List Entity → List (String × Entity) → Check (List (String × Entity))
  | [],seen => .ok seen
  | entity :: tail,seen => do
    let digest ← entityDigest p entity
    match seen.find? (fun item => item.1=digest) with
    | none => pure ()
    | some earlier => ensure (equalEntity earlier.2 entity) "conflicting lineage digest records"
    ensure (entity.parents ⊆ seen.map Prod.fst) "lineage parents already available"
    checkEntities p tail (insertSeen digest entity seen)

structure Lineage where
  schema : String
  entities : List Entity

def constructLineage (p : Primitives) (schema : String) (entities : List Entity) : Check Lineage := do
  ensure (schema="n2p3net_data_lineage/1") "data lineage schema"
  ensure (entities ≠ []) "nonempty data lineage"
  let seen ← checkEntities p entities []
  ensure (seen.length=entities.length) "lineage entities unique and topological"
  pure ⟨schema,entities⟩

def entityPayloads (p : Primitives) : List Entity → Check (List Value)
  | [] => .ok []
  | entity :: tail => do
    let payload ← entityPayload p entity
    let rest ← entityPayloads p tail
    pure (.object payload :: rest)

def lineagePayload (p : Primitives) (lineage : Lineage) : Check Fields := do
  let entities ← entityPayloads p lineage.entities
  let head ← match lineage.entities.getLast? with
    | none => .error (.fieldFailure "lineage.entities[-1]")
    | some entity => entityDigest p entity
  let metadata := Fields.ofList [("schema",.text lineage.schema),
    ("entities",.sequence (valuesOfList entities)),("entity_digest",.text head)]
  let digest ← canonicalDigest p (.object metadata)
  pure (.cons "lineage_digest" (.text digest) metadata)

def readLineage (p : Primitives) (raw : Value) : Check Lineage := do
  let fields ← objectAt raw "data lineage payload mapping"
  ensure (ExactKeys fields ["schema","entities","entity_digest","lineage_digest"]) "data lineage exact fields"
  let values ← listAt (← required fields "entities") "data lineage entity list"
  ensure (∀ value ∈ values, ∃ fields, value=.object fields) "data lineage entities are mappings"
  let entities ← readEntities p values
  let schema ← p.pythonString (← required fields "schema")
  let lineage ← constructLineage p schema entities
  let expected ← lineagePayload p lineage
  ensure (fieldDefault fields "entity_digest"=fieldDefault expected "entity_digest") "data lineage head digest"
  ensure (fieldDefault fields "lineage_digest"=fieldDefault expected "lineage_digest") "data lineage payload digest"
  pure lineage

theorem schema_gate_precedes_empty_lineage (p : Primitives) (schema : String)
    (wrong : schema ≠ "n2p3net_data_lineage/1") :
    constructLineage p schema []=.error (.sourceReject "data lineage schema") := by
  simp [constructLineage,ensure,wrong]

end
end N2P3.ERPTemporalLineage
