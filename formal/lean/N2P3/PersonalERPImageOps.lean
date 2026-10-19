import N2P3.PersonalERPInputRuntime

/-!
# Dense field assembly and explicit final-voltage QC features

Source: qc_features.py:127-187; domain.py:474-540; brainsync.py:514-623.
Array images are assembled from the actual indexed output values and masks;
no successful dataset validator is a premise. QC statistics call one named
NumPy reduction/cast at a time and then run the explicit feature validator.
-/

namespace N2P3.PersonalERPImageOps
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState N2P3.PersonalERPData
open N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime

noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def image1 {α : Type} {N : ℕ} (dtype : DType) (values : Fin N → α) : ArrayImage α :=
  ⟨dtype,[N],List.ofFn values,by simp⟩

def image2 {α : Type} {N C : ℕ} (dtype : DType) (values : Fin N → Fin C → α) : ArrayImage α :=
  ⟨dtype,[N,C],(List.ofFn (fun i => List.ofFn (values i))).flatten,by simp [Function.comp_def]⟩

def image3 {α : Type} {N C T : ℕ} (dtype : DType) (values : Fin N → Fin C → Fin T → α) : ArrayImage α :=
  ⟨dtype,[N,C,T],(List.ofFn (fun i => (List.ofFn (fun c => List.ofFn (values i c))).flatten)).flatten,
    by simp [Function.comp_def]⟩

def replaceChannels {N C T M K : ℕ} (context : PersonalERPInput.Context N C T M) (name : Value)
    (channels : List String) (values : Fin N → Fin K → Fin T → Atom) (coordinates : Fin K → Fin 3 → Atom)
    (static : Fin K → Bool) (trial : Option (Fin N → Fin K → Bool)) (quality : Option (QCFeatureImage N K))
    (provenance : Fields) (identities : Option (List PersonalERPInput.IdentityRecord)) (lineage : LineageImage) :
    PersonalERPInput.Context N K T M :=
  { data :=
      { historical := context.data.historical
        coordinateSystem := context.data.coordinateSystem
        dataAlignment := context.data.dataAlignment
        name := name
        x := image3 .floating values
        values := values
        xDecoding := rfl
        channelNames := channels
        positions := image2 .floating coordinates
        coordinates := coordinates
        positionsDecoding := rfl
        staticImage := image1 .boolean static
        static := static
        staticDecoding := rfl
        trialImage := trial.map (image2 .boolean)
        trial := trial
        trialDecoding := by cases trial <;> trivial
        subjectImage := context.data.subjectImage
        subjects := context.data.subjects
        subjectDecoding := context.data.subjectDecoding
        timeline := context.data.timeline
        labels := context.data.labels
        labelValues := context.data.labelValues
        labelDecoding := context.data.labelDecoding
        qc := quality
        metadataIsDataFrame := context.data.metadataIsDataFrame
        metadataRows := context.data.metadataRows
        metadataColumns := context.data.metadataColumns
        provenanceIsDict := true }
    preprocessingArgs := context.preprocessingArgs
    provenance := provenance
    identityCache := identities
    lineage := lineage
    attestation := none }

structure QCPrimitives where
  float64 : Atom → R Atom
  float32 : Atom → R Atom
  peakToPeak : List Atom → R Atom
  standardDeviation : List Atom → R Atom
  nanMedian : List Atom → R Atom
  divideByScaleFloor : Atom → Atom → R Atom

def castBatch (p : QCPrimitives) {N C T : ℕ} (values : Fin N → Fin C → Fin T → Atom) :
    R (Fin N → Fin C → Fin T → Atom) := do
  let rows ← (List.finRange N).mapM (fun i => (List.finRange C).mapM (fun c =>
    (List.finRange T).mapM (fun t => p.float64 (values i c t))))
  pure (fun i c t => ((rows[i.val]?.getD [])[c.val]?.getD [])[t.val]?.getD .nan)

def matrixFromRows {N C : ℕ} (rows : List (List Atom)) : Fin N → Fin C → Atom :=
  fun i c => (rows[i.val]?.getD [])[c.val]?.getD .nan

def computeQC (p : QCPrimitives) {N C T : ℕ} (values : Fin N → Fin C → Fin T → Atom)
    (static : Fin C → Bool) (trial : Option (Fin N → Fin C → Bool)) : R (QCFeatureImage N C) := do
  require (0 < N) "QC nonempty epoch array"
  require (∀ i c t, finiteNumber (values i c t)=true) "QC finite floating voltages"
  liftCheck (checkObservedMasks (N:=N) static trial)
  let observed := effectiveObserved static trial
  let doubles ← castBatch p values
  let ptpRows ← (List.finRange N).mapM (fun i => (List.finRange C).mapM (fun c =>
    p.peakToPeak (List.ofFn (doubles i c))))
  let ptp := matrixFromRows (N:=N) (C:=C) ptpRows
  let scales ← (List.finRange N).mapM (fun i => p.nanMedian
    ((List.finRange C).map (fun c => if observed i c then ptp i c else .nan)))
  let scale : Fin N → Atom := fun i => scales[i.val]?.getD .nan
  let relativeRows ← (List.finRange N).mapM (fun i => (List.finRange C).mapM (fun c =>
    p.divideByScaleFloor (ptp i c) (scale i)))
  let relative := matrixFromRows (N:=N) (C:=C) relativeRows
  let stdRows ← (List.finRange N).mapM (fun i => (List.finRange C).mapM (fun c =>
    p.standardDeviation (List.ofFn (doubles i c))))
  let std := matrixFromRows (N:=N) (C:=C) stdRows
  let relativeRows ← (List.finRange N).mapM (fun i => (List.finRange C).mapM (fun c =>
    p.float32 (if observed i c then relative i c else .real 0)))
  let stdRows ← (List.finRange N).mapM (fun i => (List.finRange C).mapM (fun c =>
    p.float32 (if observed i c then std i c else .real 0)))
  let scaleRows ← (List.finRange N).mapM (fun i => p.float32 (scale i))
  let qc : QCFeatureImage N C :=
    ⟨.text "n2p3net_epoch_qc_features/1",image2 .floating (matrixFromRows (N:=N) (C:=C) relativeRows),
      image2 .floating (matrixFromRows (N:=N) (C:=C) stdRows),image1 .floating (fun i : Fin N => scaleRows[i.val]?.getD .nan),
      image2 .boolean observed,observed,rfl⟩
  liftCheck (checkQCFeatures qc static trial)
  pure qc

theorem assembled_voltage_image_has_actual_geometry {N C T : ℕ}
    (values : Fin N → Fin C → Fin T → Atom) : (image3 .floating values).shape=[N,C,T] := rfl

theorem channel_replacement_preserves_timeline {N C T M K : ℕ} (context : PersonalERPInput.Context N C T M)
    (name : Value) (channels : List String) (values : Fin N → Fin K → Fin T → Atom)
    (coordinates : Fin K → Fin 3 → Atom) (static : Fin K → Bool) (trial : Option (Fin N → Fin K → Bool))
    (quality : Option (QCFeatureImage N K)) (provenance : Fields)
    (identities : Option (List PersonalERPInput.IdentityRecord)) (lineage : LineageImage) :
    (replaceChannels context name channels values coordinates static trial quality provenance identities lineage).data.timeline=
      context.data.timeline := rfl

end
end N2P3.PersonalERPImageOps
