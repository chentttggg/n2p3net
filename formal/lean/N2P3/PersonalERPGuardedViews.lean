import N2P3.PersonalERPArrayIngress
import N2P3.ERPTemporalArrayViews
import N2P3.PersonalERPImageOps

/-!
# Native array correspondence after the source guards

These functions only change logical coordinates: every native array cell is
re-encoded unchanged. They perform no validation, conversion, default filling,
or replacement of malformed raw input. Callers supply the shape/dtype facts
derived at their actual source guards. The Timeline comes from its actual
constructor, including its separate common-length gate; arbitrary success of
events.validate() is deliberately insufficient to construct it.
-/

namespace N2P3.PersonalERPGuardedViews
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalERPData
open N2P3.ERPTemporalNPZ N2P3.ERPTemporalArrayViews
open N2P3.PersonalERPArrayIngress N2P3.PersonalERPImageOps

noncomputable section

def flatIndex {N C : ℕ} (i : Fin N) (c : Fin C) : Fin (N*C) :=
  ⟨i.val*C+c.val,calc
    i.val*C+c.val < (i.val+1)*C := by simp [Nat.add_mul]
    _ ≤ N*C := Nat.mul_le_mul_right C i.isLt⟩

def matrix {α : Type} (cells : List α) (N C : ℕ) (length : cells.length=N*C) : Fin N → Fin C → α :=
  fun i c => vector cells (N*C) length (flatIndex i c)

theorem matrix_decoding {α : Type} (cells : List α) (N C : ℕ) (length : cells.length=N*C) :
    (List.ofFn (fun i => List.ofFn (matrix cells N C length i))).flatten=cells := by
  have split := (List.ofFn_mul (vector cells (N*C) length)).symm
  have decoded : (List.ofFn (fun i => List.ofFn (matrix cells N C length i))).flatten=
      List.ofFn (vector cells (N*C) length) := by simpa only [matrix,flatIndex] using split
  exact decoded.trans (vector_decoding cells (N*C) length)

def tensor {α : Type} (cells : List α) (N C T : ℕ) (length : cells.length=N*(C*T)) :
    Fin N → Fin C → Fin T → α :=
  fun i c t => matrix cells N (C*T) length i (flatIndex c t)

theorem tensor_decoding {α : Type} (cells : List α) (N C T : ℕ) (length : cells.length=N*(C*T)) :
    (List.ofFn (fun i => (List.ofFn (fun c => List.ofFn (tensor cells N C T length i c))).flatten)).flatten=cells := by
  unfold tensor
  have row : ∀ i : Fin N,
      (List.ofFn (fun c => List.ofFn (fun t => matrix cells N (C*T) length i (flatIndex c t)))).flatten=
        List.ofFn (matrix cells N (C*T) length i) := by
    intro i
    simpa only [flatIndex] using (List.ofFn_mul (matrix cells N (C*T) length i)).symm
  simp_rw [row]
  exact matrix_decoding cells N (C*T) length

def integerNative (raw : RawArray) (integer : raw.kind.integer=true) : DecodedArray ℤ :=
  .ofNative raw (fun z => .atom (.integer z)) (fun cell member => by
    have typed := raw.dtypeContents cell member
    cases hk : raw.kind <;> simp_all [Kind.integer,Cell.matches]
    all_goals obtain ⟨z,h⟩ := typed; exact ⟨z,h.symm⟩)

def numericNative (raw : RawArray) (numeric : raw.kind.numeric=true) : DecodedArray Atom :=
  .ofNative raw Cell.atom (fun cell member => by
    have typed := raw.dtypeContents cell member
    cases hk : raw.kind <;> simp_all [Kind.numeric,Cell.matches]
    all_goals first
      | obtain ⟨z,h⟩ := typed; exact ⟨.integer z,h.symm⟩
      | obtain ⟨a,h,_⟩ := typed; exact ⟨a,h.symm⟩
      | obtain ⟨a,h⟩ := typed; exact ⟨a,h.symm⟩)

def shapeLength {α : Type} (view : DecodedArray α) (shape : List ℕ) (h : view.raw.shape=shape) :
    view.values.length=shape.prod := view.length.trans (view.raw.storageGeometry.trans (congrArg List.prod h))

def vectorOf {α : Type} (view : DecodedArray α) (N : ℕ) (shape : view.raw.shape=[N]) : Fin N → α :=
  vector view.values N (by simpa using shapeLength view [N] shape)

def matrixOf {α : Type} (view : DecodedArray α) (N C : ℕ) (shape : view.raw.shape=[N,C]) : Fin N → Fin C → α :=
  matrix view.values N C (by simpa using shapeLength view [N,C] shape)

def tensorOf {α : Type} (view : DecodedArray α) (N C T : ℕ) (shape : view.raw.shape=[N,C,T]) :
    Fin N → Fin C → Fin T → α :=
  tensor view.values N C T (by simpa using shapeLength view [N,C,T] shape)

theorem vectorOf_decoding {α : Type} (view : DecodedArray α) (N : ℕ) (shape : view.raw.shape=[N]) :
    List.ofFn (vectorOf view N shape)=view.values := vector_decoding _ _ _

theorem matrixOf_decoding {α : Type} (view : DecodedArray α) (N C : ℕ) (shape : view.raw.shape=[N,C]) :
    (List.ofFn (fun i => List.ofFn (matrixOf view N C shape i))).flatten=view.values := matrix_decoding _ _ _ _

theorem tensorOf_decoding {α : Type} (view : DecodedArray α) (N C T : ℕ) (shape : view.raw.shape=[N,C,T]) :
    (List.ofFn (fun i => (List.ofFn (fun c => List.ofFn (tensorOf view N C T shape i c))).flatten)).flatten=view.values :=
  tensor_decoding _ _ _ _ _

abbrev LabelFields := LabelHeader

def labelImage {N : ℕ} (labels : LabelFields N) : ArrayImage Atom :=
  let decoded := integerNative labels.raw labels.dtype
  ⟨labels.raw.kind.coarse,labels.raw.shape,decoded.values.map Atom.integer,
    by simpa using decoded.length.trans labels.raw.storageGeometry⟩

def labelValues {N : ℕ} (labels : LabelFields N) : Fin N → ℤ :=
  vectorOf (integerNative labels.raw labels.dtype) N labels.shape

theorem labels_decoding {N : ℕ} (labels : LabelFields N) :
    (labelImage labels).values=List.ofFn (fun i => Atom.integer (labelValues labels i)) := by
  simpa only [labelImage,labelValues,List.map_ofFn,Function.comp_def] using
    congrArg (List.map Atom.integer) (vectorOf_decoding (integerNative labels.raw labels.dtype) N labels.shape).symm

abbrev QualityFields := QualityHeader

def qualityImage {N C : ℕ} (qc : QualityFields N C) : QCFeatureImage N C :=
  let relative := floatingArray ⟨qc.raw.relativePTP,qc.relativeDType⟩
  let std := floatingArray ⟨qc.raw.std,qc.stdDType⟩
  let scale := floatingArray ⟨qc.raw.scale,qc.scaleDType⟩
  let mask := booleanArray ⟨qc.raw.mask,qc.maskDType⟩
  ⟨qc.raw.schema,relative.image,std.image,scale.image,mask.image,
    matrixOf mask N C qc.maskShape,(matrixOf_decoding mask N C qc.maskShape).symm⟩

/-- Representation facts only, all from actual individual source guards or
actual string/timeline constructor conversions. No coordinate finiteness,
physical validity, no leakage, or entire validator-success premise is here. -/
structure Fields (raw : RawDataset) (N C T : ℕ) where
  xShape : raw.x.shape=[N,C,T]
  xDType : raw.x.kind=.floating
  positionShape : raw.positions.shape=[C,3]
  positionDType : raw.positions.kind.numeric=true
  staticShape : raw.staticMask.shape=[C]
  staticDType : raw.staticMask.kind=.boolean
  trialFields : Option {array : RawArray // array.shape=[N,C] ∧ array.kind=.boolean}
  trialCorrespondence : trialFields.map Subtype.val=raw.trialMask
  subjectCast : CastResult .string raw.subjects
  subjectShape : raw.subjects.shape=[N]
  timeline : ERPTemporalNPZ.Timeline
  timelineCorrespondence : timeline.arrays=raw.timeline.arrays
  complete : timeline.complete=raw.timeline.complete
  online : timeline.onlineCausal=raw.timeline.onlineCausal
  labels : Option (LabelFields N)
  labelCorrespondence : labels.map LabelHeader.raw=raw.labels
  quality : Option (QualityFields N C)
  qualityCorrespondence : quality.map QualityHeader.raw=raw.quality
  provenanceDict : raw.provenanceIsDict=true

def trialFields {N C : ℕ} (masks : MaskHeader N C) :
    Option {array : RawArray // array.shape=[N,C] ∧ array.kind=.boolean} :=
  match hr : masks.trialRaw with
  | none => none
  | some raw => some ⟨raw,masks.trialShape raw hr,masks.trialDType raw hr⟩

theorem trialFields_correspondence {N C : ℕ} (masks : MaskHeader N C) :
    (trialFields masks).map Subtype.val=masks.trialRaw := by
  unfold trialFields
  split <;> simp_all

/-- Successful source guards return their individual representation facts.
Only the independent, actual Timeline constructor supplies its M-domain. -/
def fieldsOfValidated (raw : RawDataset) (checked : Validated raw) (timeline : ERPTemporalNPZ.Timeline)
    (arrays : timeline.arrays=raw.timeline.arrays) (complete : timeline.complete=raw.timeline.complete)
    (online : timeline.onlineCausal=raw.timeline.onlineCausal) :
    Fields raw checked.tensor.N checked.tensor.C checked.tensor.T where
  xShape := checked.tensor.shape
  xDType := checked.tensor.floating
  positionShape := checked.positionShape
  positionDType := checked.positionDType
  staticShape := by simpa only [checked.staticCorrespondence] using checked.masks.staticShape
  staticDType := by simpa only [checked.staticCorrespondence] using checked.masks.staticDType
  trialFields := trialFields checked.masks
  trialCorrespondence := (trialFields_correspondence checked.masks).trans checked.trialCorrespondence
  subjectCast := checked.subjectCast
  subjectShape := checked.subjectShape
  timeline := timeline
  timelineCorrespondence := arrays
  complete := complete
  online := online
  labels := checked.labels
  labelCorrespondence := checked.labelCorrespondence
  quality := checked.quality
  qualityCorrespondence := checked.qualityCorrespondence
  provenanceDict := checked.provenanceDict

def context (raw : RawDataset) {N C T : ℕ} (fields : Fields raw N C T) :
    PersonalERPInput.Context N C T fields.timeline.count :=
  let x := floatingArray ⟨raw.x,fields.xDType⟩
  let positions := numericNative raw.positions fields.positionDType
  let static := booleanArray ⟨raw.staticMask,fields.staticDType⟩
  let trial := fields.trialFields.map (fun image => booleanArray ⟨image.1,image.2.2⟩)
  let trialValues := fields.trialFields.map (fun image =>
    matrixOf (booleanArray ⟨image.1,image.2.2⟩) N C image.2.1)
  let subjects := stringArray ⟨fields.subjectCast.1,fields.subjectCast.2.1⟩
  let subjectShape := fields.subjectCast.2.2.trans fields.subjectShape
  { data :=
      { historical := raw.historical
        coordinateSystem := raw.coordinates
        dataAlignment := if raw.provenanceIsMapping then raw.provenance.get "data_alignment" else none
        name := raw.name
        x := x.image
        values := tensorOf x N C T fields.xShape
        xDecoding := (tensorOf_decoding x N C T fields.xShape).symm
        channelNames := raw.channelNames
        positions := positions.image
        coordinates := matrixOf positions C 3 fields.positionShape
        positionsDecoding := (matrixOf_decoding positions C 3 fields.positionShape).symm
        staticImage := static.image
        static := vectorOf static C fields.staticShape
        staticDecoding := (vectorOf_decoding static C fields.staticShape).symm
        trialImage := trial.map DecodedArray.image
        trial := trialValues
        trialDecoding := by
          dsimp [trial,trialValues]
          cases h : fields.trialFields with
          | none => trivial
          | some image => exact (matrixOf_decoding (booleanArray ⟨image.1,image.2.2⟩) N C image.2.1).symm
        subjectImage := subjects.image
        subjects := vectorOf subjects N subjectShape
        subjectDecoding := (vectorOf_decoding subjects N subjectShape).symm
        timeline := timelineImage fields.timeline
        labels := fields.labels.map labelImage
        labelValues := fields.labels.map labelValues
        labelDecoding := by
          cases h : fields.labels with
          | none => trivial
          | some labels => exact labels_decoding labels
        qc := fields.quality.map qualityImage
        metadataIsDataFrame := raw.metadataIsDataFrame
        metadataRows := raw.metadataRows
        metadataColumns := raw.metadataColumns
        provenanceIsDict := raw.provenanceIsDict }
    preprocessingArgs := raw.preprocessingArgs
    provenance := raw.provenance
    identityCache := raw.identityCache
    lineage := raw.lineage
    attestation := raw.attestation }

theorem context_keeps_execution_receipt (raw : RawDataset) {N C T : ℕ} (fields : Fields raw N C T) :
    (context raw fields).provenance.get "preprocessing_execution"=raw.provenance.get "preprocessing_execution" := rfl

theorem context_keeps_actual_alignment_projection (raw : RawDataset) {N C T : ℕ} (fields : Fields raw N C T) :
    (context raw fields).data.dataAlignment=
      (if raw.provenanceIsMapping then raw.provenance.get "data_alignment" else none) := rfl

theorem exported_dict_is_actual_mapping (raw : RawDataset) {N C T : ℕ} (fields : Fields raw N C T) :
    raw.provenanceIsMapping=true := raw.dictImpliesMapping fields.provenanceDict

/-- Full raw dataset checking feeds the actual post-dataset input continuation
once. No entire validator/constructor success is supplied as a premise. -/
def validateInput (p : Operations) (metadata : N2P3.FrozenState.Fields) (basis : Option N2P3.ERPState.TemporalModel)
    (raw : RawDataset) (timeline : ERPTemporalNPZ.Timeline)
    (arrays : timeline.arrays=raw.timeline.arrays) (complete : timeline.complete=raw.timeline.complete)
    (online : timeline.onlineCausal=raw.timeline.onlineCausal) :
    PersonalERPScalar.Result (Σ N C T, PersonalERPInput.Context N C T timeline.count) := do
  let checked ← PersonalERPArrayIngress.validate p raw false
  let fields := fieldsOfValidated raw checked timeline arrays complete online
  let target ← PersonalERPInputRuntime.validateAfterDataset p.input metadata basis (context raw fields)
  pure ⟨checked.tensor.N,checked.tensor.C,checked.tensor.T,target⟩

end
end N2P3.PersonalERPGuardedViews
