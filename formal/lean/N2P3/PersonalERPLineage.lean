import N2P3.PersonalERPImageOps

/-!
# Literal source lineage materialization and derived-entity construction

Source: lineage.py:14-210; epochs.py:729-762; events.py:488-524.
Parameters, parents, semantic records, source provenance selection and event
fingerprint rows are constructed explicitly. Only string normalization and a
single semantic hash/JSON encoder are operations; no lineage-validity or
projection-authorization oracle is supplied.
-/

namespace N2P3.PersonalERPLineage
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPModels N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Entity where
  operation : String
  parents : List String
  parameters : Fields

structure Lineage where
  entities : List Entity
  schema : String := "n2p3net_data_lineage/1"

structure Frame (N C T M : ℕ) where
  context : PersonalERPInput.Context N C T M
  timing : PersonalERPRaw.TimingTable
  timingRows : timing.rows.length=context.data.metadataRows
  timingColumns : timing.columns.length=context.data.metadataColumns
  timingSource : String
  lineage : Option Lineage
  valueDType : String
  positionDType : String

structure Primitives where
  input : PersonalERPInput.Primitives
  lower : String → String
  fingerprintHash : Value → R String

def keysNonempty : Value → Prop
  | .object fields => fieldsNonempty fields
  | .sequence values => valuesNonempty values
  | _ => True
where
  fieldsNonempty : Fields → Prop
    | .nil => True
    | .cons key value tail => key ≠ "" ∧ keysNonempty value ∧ fieldsNonempty tail
  valuesNonempty : Values → Prop
    | .nil => True
    | .cons value tail => keysNonempty value ∧ valuesNonempty tail

def semanticEntity (entity : Entity) : Fields := Fields.ofList
  [("operation",.text entity.operation),("parent_entity_digests",.sequence (FrozenState.encodeTexts entity.parents)),
   ("parameters",.object entity.parameters)]

def entityDigest (p : Primitives) (entity : Entity) : R String := do
  require (keysNonempty (.object entity.parameters)) "lineage canonical nonempty mapping keys"
  liftCheck (p.input.execution.semanticHash (.object (semanticEntity entity)))

def constructEntity (p : Primitives) (operation : String) (parents : List String) (parameters : Fields) : R Entity := do
  let operation := p.input.execution.strip operation
  require (operation ≠ "") "lineage nonempty operation"
  let parents := (parents.map (fun value => p.lower (p.input.execution.strip value))).dedup.mergeSort (· ≤ ·)
  require (∀ parent ∈ parents, StateLedger.LowerHex64 parent) "lineage parent SHA digest syntax"
  require (operation ≠ "source_ingress" ∨ parents=[]) "source lineage has no parents"
  require (operation="source_ingress" ∨ parents ≠ []) "derived lineage requires parents"
  require (parameters ≠ .nil) "lineage nonempty parameters mapping"
  require (keysNonempty (.object parameters)) "lineage canonical parameter keys"
  pure ⟨operation,parents,parameters⟩

def sameEntity (first second : Entity) : Prop :=
  first.operation=second.operation ∧ first.parents=second.parents ∧
    PersonalAdmission.PyEq (.object first.parameters) (.object second.parameters)

def checkEntities (p : Primitives) : List Entity → List (String × Entity) → R (List (String × Entity))
  | [],seen => .ok seen
  | entity :: tail,seen => do
    let digest ← entityDigest p entity
    match seen.lookup digest with
    | none => pure ()
    | some previous => require (sameEntity previous entity) "lineage no conflicting digest records"
    require (entity.parents ⊆ seen.map Prod.fst) "lineage parents available in topological order"
    checkEntities p tail (seen ++ [(digest,entity)])

def constructLineage (p : Primitives) (entities : List Entity) (schema : String := "n2p3net_data_lineage/1") : R Lineage := do
  require (schema="n2p3net_data_lineage/1") "lineage schema one"
  require (entities ≠ []) "lineage at least one entity"
  let seen ← checkEntities p entities []
  require ((seen.map Prod.fst).Nodup) "lineage unique ordered entities"
  pure ⟨entities,schema⟩

def source (p : Primitives) (parameters : Fields) : R Lineage := do
  let entity ← constructEntity p "source_ingress" [] parameters
  constructLineage p [entity]

def headDigest (p : Primitives) (lineage : Lineage) : R String := do
  let entity ← match lineage.entities.getLast? with
    | none => .error (.inherited (.fieldFailure "lineage head entity index"))
    | some entity => .ok entity
  entityDigest p entity

def mergeParent (p : Primitives) : List Entity → List (String × Entity) → R (List (String × Entity))
  | [],merged => .ok merged
  | entity :: tail,merged => do
    let digest ← entityDigest p entity
    let merged ← match merged.lookup digest with
      | none => pure (merged ++ [(digest,entity)])
      | some previous => do
        require (sameEntity previous entity) "parent lineages agree for one digest"
        pure merged
    mergeParent p tail merged

def derive (p : Primitives) (parents : List Lineage) (operation : String) (parameters : Fields) : R Lineage := do
  require (parents ≠ []) "derived lineage requires parent lineages"
  let merged ← parents.foldlM (fun merged parent => mergeParent p parent.entities merged) []
  let heads ← parents.mapM (headDigest p)
  let entity ← constructEntity p operation heads parameters
  let digest ← entityDigest p entity
  require (digest ∉ merged.map Prod.fst) "derived operation does not duplicate existing entity"
  constructLineage p ((merged.map Prod.snd) ++ [entity])

def fingerprintRows {M : ℕ} (timeline : TimelineImage M) : Value :=
  .sequence (valuesOfList ((List.finRange M).map (fun i => .sequence (valuesOfList
    [.text (timeline.eventIds i),.text (timeline.groups i),.text (timeline.subjects i),
     .integer (timeline.stimulus i),.integer (timeline.onsetSamples i),
     PersonalERPQuality.finiteFloatJSON (timeline.onset i),
     PersonalERPQuality.finiteFloatJSON (timeline.availableAt i),.integer (timeline.evidence i),
     .text (timeline.statuses i),.text (timeline.details i),.text (timeline.datasets i),
     .text (timeline.sessions i),.text (timeline.runs i),.text (timeline.selections i),
     .text (timeline.candidates i),.text (timeline.targets i),.integer (timeline.repetitions i)]))))

def fingerprint (p : Primitives) {N C T M : ℕ} (frame : Frame N C T M) : R String := do
  liftCheck (checkTimeline p.input.execution.strip frame.context.data.timeline N)
  p.fingerprintHash (.object (Fields.ofList [("schema",.text "n2p3net_scheduled_events/2"),
    ("complete",.boolean frame.context.data.timeline.complete),
    ("online_causal",.boolean frame.context.data.timeline.onlineCausal),("timing_source",.text frame.timingSource),
    ("events",fingerprintRows frame.context.data.timeline),("truth",.sequence .nil)]))

def stableProvenance (fields : Fields) : Fields :=
  Fields.ofList ((fieldPairs fields).filter (fun pair => decide
    (pair.1 ∈ ["source","source_reference","source_sample_rate_hz","session_schema","preprocessing_execution"] ∨
      (pair.1.toList.reverse.take 7).reverse= "_sha256".toList)))

def materialize (p : Primitives) {N C T M : ℕ} (frame : Frame N C T M) : R (Lineage × Frame N C T M) := do
  match frame.lineage with
  | some lineage => pure (lineage,frame)
  | none =>
    let identity ← liftCheck (PersonalERPInput.materializeIdentity p.input frame.context)
    let context := {frame.context with identityCache:=some identity.1}
    liftCheck (checkTimeline p.input.execution.strip context.data.timeline N)
    let frame := {frame with context:=context}
    let eventFingerprint ← fingerprint p frame
    let identityDigest ← liftCheck (p.input.execution.semanticHash
      (.object (PersonalERPInput.identityCanonicalPayload identity.1)))
    let args ← liftCheck (readPreprocessingArgs context.preprocessingArgs)
    let datasets := (List.ofFn context.data.timeline.datasets).dedup.mergeSort (· ≤ ·)
    let parameters := Fields.ofList [("dataset_name",context.data.name),
      ("source_dataset_ids",.sequence (FrozenState.encodeTexts datasets)),
      ("event_fingerprint",.text eventFingerprint),("identity_digest",.text identityDigest),
      ("preprocessing",.object (preprocessingRecord args)),("shape",.sequence (FrozenState.encodeNats [N,C,T])),
      ("source_provenance",.object (stableProvenance context.provenance))]
    let lineage ← source p parameters
    let context := {context with lineage:=.dataLineage}
    pure (lineage,{frame with context:=context,lineage:=some lineage})

theorem existing_lineage_is_used_without_reconstruction (p : Primitives) {N C T M : ℕ}
    (frame : Frame N C T M) (lineage : Lineage) (h : frame.lineage=some lineage) :
    materialize p frame=.ok (lineage,frame) := by simp [materialize,h]

end
end N2P3.PersonalERPLineage
