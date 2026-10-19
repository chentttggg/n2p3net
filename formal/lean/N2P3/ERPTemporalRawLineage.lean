import N2P3.ERPTemporalRawFields
import N2P3.ERPTemporalLineage

/-! Raw JSON lineage arguments remain raw until lineage's own _canonical
rejects nonfinite parameter values. Other constructor str operations accept
their original raw values. The normalized finite Entity is built only after
those source gates; later entity/topological/head/digest bodies stay explicit. -/

namespace N2P3.ERPTemporalRawLineage

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.ERPTemporalJSON N2P3.ERPTemporalLineage

noncomputable section

attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Primitives where
  pythonString : JValue → Check String
  strip : String → String
  lower : String → String
  semanticHash : N2P3.FrozenState.Value → Check String

def finitePrimitives (p : Primitives) : ERPTemporalLineage.Primitives :=
  ⟨(fun value => p.pythonString (fromValue value)),p.strip,p.lower,p.semanticHash⟩

def constructEntity (p : Primitives) (operation : String) (parents : List String)
    (parameters : JFields) : Check Entity := do
  let operation := p.strip operation
  ensure (operation ≠ "") "lineage operation nonempty"
  let parents ← parentDigests (finitePrimitives p) parents
  let parents := parents.dedup.mergeSort (· ≤ ·)
  ensure (operation="source_ingress" → parents=[]) "source lineage has no parents"
  ensure (operation ≠ "source_ingress" → parents ≠ []) "derived lineage needs parents"
  ensure (parameters.keys ≠ []) "lineage parameters nonempty mapping"
  let parameters ← ERPTemporalRawFields.canonicalLineageFields parameters
  pure ⟨operation,parents,parameters⟩

def readEntity (p : Primitives) (raw : JValue) : Check Entity := do
  let fields ← ERPTemporalRawFields.objectAt raw "lineage entity mapping"
  ensure (ERPTemporalRawFields.exactFields fields ["operation","parent_entity_digests","parameters","entity_digest"])
    "lineage entity exact fields"
  let parents ← ERPTemporalRawFields.stringList
    (← ERPTemporalRawFields.required fields "parent_entity_digests") "lineage parents string list"
  let parameters ← ERPTemporalRawFields.objectAt
    (← ERPTemporalRawFields.required fields "parameters") "lineage parameters mapping"
  let operation ← p.pythonString (← ERPTemporalRawFields.required fields "operation")
  let entity ← constructEntity p operation parents parameters
  let digest ← entityDigest (finitePrimitives p) entity
  ensure (pythonEqual (ERPTemporalRawFields.fieldDefault fields "entity_digest") (.text digest)) "lineage entity digest"
  pure entity

def readEntities (p : Primitives) : List JValue → Check (List Entity)
  | [] => .ok []
  | value :: tail => do
    let entity ← readEntity p value
    let tail ← readEntities p tail
    pure (entity :: tail)

def readLineage (p : Primitives) (raw : JValue) : Check Lineage := do
  let fields ← ERPTemporalRawFields.objectAt raw "data lineage payload mapping"
  ensure (ERPTemporalRawFields.exactFields fields ["schema","entities","entity_digest","lineage_digest"])
    "data lineage exact fields"
  let values ← ERPTemporalRawFields.listAt (← ERPTemporalRawFields.required fields "entities") "lineage entities list"
  ensure (∀ value ∈ values, ∃ fields, value=.object fields) "lineage entities are mappings"
  let entities ← readEntities p values
  let schema ← p.pythonString (← ERPTemporalRawFields.required fields "schema")
  let lineage ← ERPTemporalLineage.constructLineage (finitePrimitives p) schema entities
  let expected ← lineagePayload (finitePrimitives p) lineage
  ensure (pythonEqual (ERPTemporalRawFields.fieldDefault fields "entity_digest")
    (fromValue (N2P3.StateLedger.fieldDefault expected "entity_digest"))) "data lineage head digest"
  ensure (pythonEqual (ERPTemporalRawFields.fieldDefault fields "lineage_digest")
    (fromValue (N2P3.StateLedger.fieldDefault expected "lineage_digest"))) "data lineage payload digest"
  pure lineage

end
end N2P3.ERPTemporalRawLineage
