import N2P3.ERPTemporalRawFields

/-! identity.py:149-177,186-198,251-269. Preserve from_payload's origin/global
list type gates BEFORE constructor arguments str(local)/str(status), then the
constructor's own local/origin/global cleaning and status str. No finite-JSON
projection is inserted before these actual guards. -/

namespace N2P3.ERPTemporalRawIdentity

open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPInput
open N2P3.ERPTemporalJSON

noncomputable section

attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Primitives where
  input : PersonalERPInput.Primitives
  pythonString : JValue → Check String

def cleanText (p : Primitives) (value : String) : Check String := do
  PersonalERPInput.cleanString p.input (← p.pythonString (.text value))

def cleanTexts (p : Primitives) : List String → Check (List String)
  | [] => .ok []
  | value :: tail => do
    let value ← cleanText p value
    let tail ← cleanTexts p tail
    pure (value :: tail)

def constructRecord (p : Primitives) (localSubject status : String)
    (origins globals : List String) : Check IdentityRecord := do
  let localSubject ← cleanText p localSubject
  let origins ← cleanTexts p origins
  let globals ← cleanTexts p globals
  let origins := origins.dedup.mergeSort (· ≤ ·)
  let globals := globals.dedup.mergeSort (· ≤ ·)
  let status ← p.pythonString (.text status)
  ensure (status ∈ ["source_verified","global_verified","cross_study_unknown"]) "identity status enum"
  ensure (status="source_verified" → origins ≠ []) "source identity needs origins"
  ensure (status="global_verified" → origins ≠ [] ∧ globals ≠ []) "global identity needs source and global keys"
  ensure (globals ≠ [] → status="global_verified") "global keys require verified status"
  ensure (status="cross_study_unknown" → origins=[] ∧ globals=[]) "unknown identity has no authority keys"
  pure ⟨localSubject,origins,globals,status⟩

def readRecord (p : Primitives) (raw : JValue) : Check IdentityRecord := do
  let fields ← ERPTemporalRawFields.objectAt raw "participant identity mapping"
  ensure (ERPTemporalRawFields.exactFields fields
    ["local_subject_id","origin_subject_keys","global_person_keys","identity_status"])
    "participant identity exact fields"
  let originsRaw ← ERPTemporalRawFields.required fields "origin_subject_keys"
  let globalsRaw ← ERPTemporalRawFields.required fields "global_person_keys"
  let origins ← ERPTemporalRawFields.stringList originsRaw "origin keys list of strings"
  let globals ← ERPTemporalRawFields.stringList globalsRaw "global keys list of strings"
  let localSubject ← p.pythonString (← ERPTemporalRawFields.required fields "local_subject_id")
  let status ← p.pythonString (← ERPTemporalRawFields.required fields "identity_status")
  constructRecord p localSubject status origins globals

def readRecords (p : Primitives) : List JValue → Check (List IdentityRecord)
  | [] => .ok []
  | value :: tail => do
    let record ← readRecord p value
    let tail ← readRecords p tail
    pure (record :: tail)

def readTable (p : Primitives) (raw : JValue) : Check (List IdentityRecord) := do
  let fields ← ERPTemporalRawFields.objectAt raw "dataset identity payload mapping"
  ensure (ERPTemporalRawFields.exactFields fields ["schema","records","digest"])
    "dataset identity exact fields"
  let values ← ERPTemporalRawFields.listAt (← ERPTemporalRawFields.required fields "records") "identity records list"
  ensure (∀ value ∈ values, ∃ fields, value=.object fields) "identity records are mappings"
  let records ← readRecords p values
  let schema ← p.pythonString (← ERPTemporalRawFields.required fields "schema")
  ensure (schema="n2p3net_dataset_identity_table/1") "dataset identity schema"
  let records := records.mergeSort (fun a b => a.localSubject ≤ b.localSubject)
  ensure (records ≠ []) "nonempty dataset identity table"
  ensure ((records.map IdentityRecord.localSubject).Nodup) "unique dataset identity local subjects"
  let digest ← p.input.execution.semanticHash (.object (PersonalERPInput.identityCanonicalPayload records))
  ensure (pythonEqual (ERPTemporalRawFields.fieldDefault fields "digest") (.text digest)) "dataset identity canonical digest"
  pure records

end
end N2P3.ERPTemporalRawIdentity
