import N2P3.PersonalERPLineage
import N2P3.PersonalERPGeometry

/-!
# Actual post-epoch measured-reference projection and identity branches

Source: brainsync.py:132-150; domain.py:400-545. The source coordinate/mask
checks, existing-projection lineage witness and exact source-name lookup are
kept. New values are cast to float32 before in-place subtraction of the
original reference signal; its source dtype is retained as an operand tag.
QC features, identity cache, source lineage and the derived lineage are built
in source order, then the actual output dataset validator is invoked.
-/

namespace N2P3.PersonalERPPostReference
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPModels N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime
open N2P3.PersonalERPImageOps N2P3.PersonalERPLineage

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure Primitives where
  input : PersonalERPExecutionRuntime.Primitives
  geometry : PersonalERPGeometry.Primitives
  lineage : PersonalERPLineage.Primitives
  qc : QCPrimitives
  float32 : Atom → R Atom
  subtractInplace32 : String → Atom → Atom → R Atom
  positionsClose : String → ArrayImage Atom → List PersonalERPGeometry.Vector → R Bool
  positionClose : String → (Fin 3 → Atom) → PersonalERPGeometry.Vector → R Bool

abbrev Packed (N T M : ℕ) := Σ C, Frame N C T M

def actualMask {N C T M : ℕ} (frame : Frame N C T M) : Fin N → Fin C → Bool :=
  frame.context.data.trial.getD (fun _ c => frame.context.data.static c)

def sourceIndex {N C T M : ℕ} (frame : Frame N C T M) (name : String) : Option (Fin C) :=
  ((frame.context.data.channelNames.zip (List.finRange C)).reverse.lookup name)

def standard (p : Primitives) (requested : List String) : R (List String × List PersonalERPGeometry.Vector) := do
  let channels ← liftCheck (PersonalERPGeometry.canonicalUnique p.geometry requested)
  let positions ← liftCheck (PersonalERPGeometry.standardPositions p.geometry channels)
  let _ ← liftCheck (p.geometry.embedding positions)
  pure (channels,positions)

def previousReferenceMatches (p : Primitives) (reference : Fields) (anchor : Option String) : R Bool := do
  match anchor with
  | some channel => pure (decide (PersonalAdmission.PyEq (.object reference)
      (.object (Fields.ofList [("mode",.text "measured_channel"),("channel",.text channel)]))))
  | none =>
    let nonempty := match fieldDefault reference "source_reference" with
      | .text value => decide (p.input.data.input.original.execution.strip value ≠ "")
      | _ => false
    pure (decide (fieldDefault reference "mode"=.text "native_preserve") && nonempty)

def existingProjection (p : Primitives) {N C T M : ℕ} (frame : Frame N C T M) (previous : Value)
    (channels : List String) (positions : List PersonalERPGeometry.Vector) (anchor : Option String) (name : Value) :
    R (Frame N C T M) := do
  let previousFields := match previous with | .object fields => fields | _ => .nil
  let referenceValue := (previousFields.get "reference").getD (.object .nil)
  let reference ← PersonalERPModelRuntime.objectAt referenceValue "existing montage reference mapping"
  let referenceMatches ← previousReferenceMatches p reference anchor
  let expected ← match anchor with
    | some anchor => pure ("measured_channel:" ++ anchor)
    | none => do
      let source ← liftCheck (p.input.data.input.original.pythonString (fieldDefault reference "source_reference"))
      pure ("native:" ++ source)
  require (∃ fields, previous=.object fields) "existing projection must be mapping"
  require (fieldDefault previousFields "schema"=.text "n2p3_observed_montage/1") "existing projection schema"
  require (PersonalAdmission.PyEq (fieldDefault previousFields "target_channels") (.sequence (encodeTexts channels))) "existing projection target order"
  require (fieldDefault previousFields "operation"=.text "select_reference_and_observation_mask") "existing projection operation"
  let args ← liftCheck (readPreprocessingArgs frame.context.preprocessingArgs)
  require (PersonalAdmission.PyEq (fieldDefault previousFields "preprocessing_identity") (fieldDefault args "name")) "existing projection preprocessing name"
  require (referenceMatches=true) "existing projection matching reference"
  require (frame.context.data.channelNames=channels) "existing projection actual channel names"
  let close ← p.positionsClose frame.positionDType frame.context.data.positions positions
  require (close=true) "existing projection standard coordinates"
  require (fieldDefault frame.context.provenance "source_reference"=.text expected) "existing projection actual reference label"
  let materialized ← PersonalERPLineage.materialize p.lineage frame
  require (∃ entity ∈ materialized.1.entities, entity.operation="observed_montage_projection" ∧
    PersonalAdmission.PyEq (.object entity.parameters) (.object previousFields)) "existing projection actual lineage witness"
  if name ≠ .null then require (PersonalAdmission.PyEq name frame.context.data.name) "existing projection cannot be renamed"
  pure materialized.2

def channelRows (p : Primitives) {N C T M : ℕ} (frame : Frame N C T M)
    (anchor : Option (Fin C)) : List (String × PersonalERPGeometry.Vector) →
    R (List (List (List Atom)) × List (List Bool))
  | [] => .ok ([],[])
  | (name,position) :: tail => do
    let row ← match sourceIndex frame name with
      | none => pure ((List.finRange N).map (fun _ => List.replicate T (Atom.real 0)),List.replicate N false)
      | some source => do
        if ∃ i, actualMask frame i source=true then
          let close ← p.positionClose frame.positionDType (frame.context.data.coordinates source) position
          require (close=true) "observed source matches declared standard coordinates"
        let values ← (List.finRange N).mapM (fun i => do
          -- Source assignment/cast precedes its in-place reference subtraction.
          let assigned ← (List.finRange T).mapM (fun t => p.float32 (frame.context.data.values i source t))
          match anchor with
          | none => pure assigned
          | some anchor => do
              (assigned.zip (List.ofFn (frame.context.data.values i anchor))).mapM
                (fun pair => p.subtractInplace32 frame.valueDType pair.1 pair.2))
        pure (values,List.ofFn (fun i => actualMask frame i source))
    let rest ← channelRows p frame anchor tail
    pure (row.1::rest.1,row.2::rest.2)

def projectNew (p : Primitives) {N C T M : ℕ} (frame : Frame N C T M) (channels : List String)
    (positions : List PersonalERPGeometry.Vector) (anchor : Option String) (name : Value) : R (Packed N T M) := do
  let reference ← PersonalERPModelRuntime.text (fieldDefault frame.context.provenance "source_reference") "explicit native reference string"
  require (p.input.data.input.original.execution.strip reference ≠ "") "nonempty native reference"
  let anchorIndex ← match anchor with
    | none => pure none
    | some anchor => do
      let index ← match sourceIndex frame anchor with
        | none => .error (.sourceValueError "measured anchor in source channel mapping")
        | some index => .ok index
      require (∀ i, actualMask frame i index=true) "measured anchor observed in every trial"
      pure (some index)
  -- standard() maps one position for every requested channel; the zip is a
  -- representation of those same ordered arrays, never a free row bijection.
  let rows ← channelRows p frame anchorIndex (channels.zip positions)
  let K := channels.length
  let observed : Fin N → Fin K → Bool := fun i c => (rows.2[c.val]?.getD [])[i.val]?.getD false
  let values : Fin N → Fin K → Fin T → Atom := fun i c t =>
    if observed i c then (((rows.1[c.val]?.getD [])[i.val]?.getD [])[t.val]?.getD .nan) else .real 0
  require (∀ i, ∃ c, observed i c=true) "each projected trial retains measured EEG"
  let static : Fin K → Bool := fun c => decide (∃ i, observed i c=true)
  let coordinates : Fin K → Fin 3 → Atom := fun c axis =>
    .real ((positions[c.val]?.getD (fun _ => 0)) axis)
  let referenceRecord := match anchor with
    | none => Fields.ofList [("mode",.text "native_preserve"),("source_reference",.text reference)]
    | some anchor => Fields.ofList [("mode",.text "measured_channel"),("channel",.text anchor)]
  let outputReference := match anchor with | none => "native:" ++ reference | some anchor => "measured_channel:" ++ anchor
  let args ← liftCheck (readPreprocessingArgs frame.context.preprocessingArgs)
  let adapter := Fields.ofList [("schema",.text "n2p3_observed_montage/1"),
    ("target_channels",.sequence (encodeTexts channels)),("operation",.text "select_reference_and_observation_mask"),
    ("reference",.object referenceRecord),("preprocessing_identity",fieldDefault args "name")]
  let sourceName ← PersonalERPModelRuntime.text frame.context.data.name "projected source dataset name"
  let outputName := if PyTruthy name then name else .text (sourceName ++ "-masked-montage")
  let qc ← computeQC p.qc values static (some observed)
  let identity ← liftCheck (PersonalERPInput.materializeIdentity p.input.data.input.original frame.context)
  let parentContext := {frame.context with identityCache:=some identity.1}
  let parent := {frame with context:=parentContext}
  let materialized ← PersonalERPLineage.materialize p.lineage parent
  let lineage ← PersonalERPLineage.derive p.lineage [materialized.1] "observed_montage_projection" adapter
  let missing := ((List.finRange K).filter (fun c => !(static c))).map (fun c => channels[c.val]?.getD "")
  let additions := [("source_reference",.text outputReference),("parent_source_reference",.text reference),
    ("domain_adapter",.object adapter),("source_acquisition_channels",.sequence (encodeTexts frame.context.data.channelNames)),
    ("missing_target_channels",.sequence (encodeTexts missing)),("unobserved_values_are_placeholders",.boolean true)]
  let provenance := additions.foldl (fun fields pair => Fields.cons pair.1 pair.2 (fields.erase pair.1)) frame.context.provenance
  let output := replaceChannels frame.context outputName channels values coordinates static (some observed) (some qc)
    provenance (some identity.1) .dataLineage
  PersonalERPInputRuntime.checkDataset p.input output frame.context.data.labels.isSome
  pure ⟨K,⟨output,frame.timing,frame.timingRows,frame.timingColumns,frame.timingSource,some lineage,"float32","float32"⟩⟩

def adapt (p : Primitives) {N C T M : ℕ} (frame : Frame N C T M) (requested : List String)
    (reference : Option String) (name : Value := .null) : R (Packed N T M) := do
  PersonalERPInputRuntime.checkDataset p.input frame.context frame.context.data.labels.isSome
  let (channels,positions) ← standard p requested
  let anchor ← match reference with
    | none => pure none
    | some reference => (liftCheck (p.input.data.input.original.canonicalChannel reference)).map some
  if anchor.isSome then
    let ocular ← PersonalERPModelRuntime.objectAt
      ((frame.context.provenance.get "ocular_artifact_processing").getD (.object .nil)) "projection ocular processing.get mapping"
    require (fieldDefault ocular "status" ≠ .text "applied") "reference must precede ocular regression"
    let args ← liftCheck (readPreprocessingArgs frame.context.preprocessingArgs)
    require (fieldDefault args "baseline_mode" ≠ .text "trial_reference" ∨
      fieldDefault args "trial_reference_center"=.text "mean") "measured projection linear baseline"
  require (2 ≤ channels.length) "at least two declared masked montage channels"
  match frame.context.provenance.get "domain_adapter" with
  | some previous => do
    if previous=.null then
      projectNew p frame channels positions anchor name
    else
      let existing ← existingProjection p frame previous channels positions anchor name
      pure ⟨C,existing⟩
  | none => projectNew p frame channels positions anchor name

def apply (p : Primitives) {N C T M : ℕ} (frame : Frame N C T M) (reference : Option String)
    (ocular : Option EOGModel) (channels : Option (List String)) : R (Packed N T M) := do
  match reference with
  | none => pure ⟨C,frame⟩
  | some reference => do
    let anchor ← liftCheck (p.input.data.input.original.canonicalChannel reference)
    match ocular with
    | some ocular =>
      require (ocular.reference=some anchor) "requested reference agrees with frozen ocular model"
      pure ⟨C,frame⟩
    | none =>
      let continuous ← PersonalERPModelRuntime.objectAt
        ((frame.context.provenance.get "continuous_artifact_processing").getD (.object .nil)) "post reference continuous processing mapping"
      let model ← PersonalERPModelRuntime.objectAt ((continuous.get "model").getD (.object .nil)) "post reference continuous model mapping"
      let spec ← PersonalERPModelRuntime.objectAt ((model.get "spec").getD (.object .nil)) "post reference continuous spec mapping"
      let qcAnchor := fieldDefault spec "reference_channel"
      if qcAnchor ≠ .null then
        require (qcAnchor=.text anchor ∧ fieldDefault frame.context.provenance "source_reference"=.text ("measured_channel:" ++ anchor))
          "requested reference agrees with frozen QC reference"
        match channels with
        | none => pure ()
        | some channels => require (channels=frame.context.data.channelNames) "QC final montage equals saved ordered request"
        pure ⟨C,frame⟩
      else
        let selected ← match channels with
          | some channels => if channels ≠ [] then pure channels else do
              let canonical ← liftCheck (frame.context.data.channelNames.mapM p.input.data.input.original.canonicalChannel)
              pure (((frame.context.data.channelNames.zip canonical).filter (fun row => decide (row.2 ≠ anchor))).map Prod.fst)
          | none => do
            let canonical ← liftCheck (frame.context.data.channelNames.mapM p.input.data.input.original.canonicalChannel)
            pure (((frame.context.data.channelNames.zip canonical).filter (fun row => decide (row.2 ≠ anchor))).map Prod.fst)
        adapt p frame selected (some anchor)

theorem no_reference_returns_before_all_projection_checks (p : Primitives) {N C T M : ℕ}
    (frame : Frame N C T M) (ocular : Option EOGModel) (channels : Option (List String)) :
    apply p frame none ocular channels=.ok ⟨C,frame⟩ := rfl

end
end N2P3.PersonalERPPostReference
