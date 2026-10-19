import N2P3.ERPTemporalCacheDecode

/-!
# Coordinate views derived from individual native array images

These are logical projections of ndarray dtype/storage, not new source
validation calls. Decoding keeps the complete raw shape and every cell. No
N/C/T view is constructed for an EpochDataset here: raw X/subjects/masks can
still have inconsistent lengths and the loader/CLI retains those states.
Only Timeline's OWN constructor has already checked common lengths; its M
coordinate domain can therefore be constructed at this stage.
-/

namespace N2P3.ERPTemporalArrayViews

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalERPData
open N2P3.ERPTemporalNPZ

noncomputable section

def viewCells {α : Type} (encode : α → Cell) : (cells : List Cell) →
    (∀ cell ∈ cells, ∃ value, encode value=cell) → List α
  | [],_ => []
  | cell :: tail,complete =>
    Classical.choose (complete cell (by simp)) ::
      viewCells encode tail (fun value member => complete value (by simp [member]))

theorem viewCells_reencode {α : Type} (encode : α → Cell) (cells : List Cell)
    (complete : ∀ cell ∈ cells, ∃ value, encode value=cell) :
    (viewCells encode cells complete).map encode=cells := by
  induction cells with
  | nil => rfl
  | cons cell tail ih =>
    simp only [viewCells,List.map_cons,Classical.choose_spec (complete cell (by simp)),ih]

structure DecodedArray (α : Type) where
  raw : RawArray
  encode : α → Cell
  values : List α
  decoding : values.map encode=raw.cells

def DecodedArray.ofNative {α : Type} (raw : RawArray) (encode : α → Cell)
    (complete : ∀ cell ∈ raw.cells, ∃ value, encode value=cell) : DecodedArray α :=
  ⟨raw,encode,viewCells encode raw.cells complete,viewCells_reencode encode raw.cells complete⟩

theorem DecodedArray.length {α : Type} (view : DecodedArray α) :
    view.values.length=view.raw.cells.length := by
  have eq := congrArg List.length view.decoding
  simpa only [List.length_map] using eq

def DecodedArray.image {α : Type} (view : DecodedArray α) : ArrayImage α :=
  ⟨view.raw.kind.coarse,view.raw.shape,view.values,view.length.trans view.raw.storageGeometry⟩

def stringArray (image : {raw : RawArray // raw.kind=.unicode}) : DecodedArray String :=
  .ofNative image.1 Cell.text (fun cell member => by
    have typed := image.1.dtypeContents cell member
    rw [image.2] at typed
    obtain ⟨value,equal⟩ := typed
    exact ⟨value,equal.symm⟩)

def integerArray (image : {raw : RawArray // raw.kind=.signed}) : DecodedArray ℤ :=
  .ofNative image.1 (fun value => .atom (.integer value)) (fun cell member => by
    have typed := image.1.dtypeContents cell member
    rw [image.2] at typed
    obtain ⟨value,equal⟩ := typed
    exact ⟨value,equal.symm⟩)

def booleanArray (image : {raw : RawArray // raw.kind=.boolean}) : DecodedArray Bool :=
  .ofNative image.1 (fun value => .atom (.boolean value)) (fun cell member => by
    have typed := image.1.dtypeContents cell member
    rw [image.2] at typed
    obtain ⟨value,equal⟩ := typed
    exact ⟨value,equal.symm⟩)

def floatingArray (image : {raw : RawArray // raw.kind=.floating}) : DecodedArray Atom :=
  .ofNative image.1 Cell.atom (fun cell member => by
    have typed := image.1.dtypeContents cell member
    rw [image.2] at typed
    obtain ⟨value,equal,_⟩ := typed
    exact ⟨value,equal.symm⟩)

def coordinate {α : Type} {M : ℕ} (values : List α) (length : values.length=M) (i : Fin M) : α :=
  values.get ⟨i.val,by simpa only [length] using i.isLt⟩

theorem coordinate_reencodes_original_cell {α : Type} (view : DecodedArray α)
    (i : Fin view.raw.cells.length) :
    view.encode (coordinate view.values view.length i)=view.raw.cells.get i := by
  have mapped := congrArg (fun values : List Cell => values[i.val]?) view.decoding
  have within : i.val < view.values.length := by simpa only [view.length] using i.isLt
  simpa only [List.getElem?_map,List.getElem?_eq_getElem within,List.getElem?_eq_getElem i.isLt,
    Option.map_some,Option.some.injEq,coordinate,List.get_eq_getElem] using mapped

def timelineStrings (timeline : ERPTemporalNPZ.Timeline) (field : Fin 17)
    (string : (eventCast field).kind=.unicode) : DecodedArray String :=
  stringArray ⟨timeline.arrays field,(timeline.kinds field).trans string⟩

def timelineIntegers (timeline : ERPTemporalNPZ.Timeline) (field : Fin 17)
    (integer : (eventCast field).kind=.signed) : DecodedArray ℤ :=
  integerArray ⟨timeline.arrays field,(timeline.kinds field).trans integer⟩

def timelineFloats (timeline : ERPTemporalNPZ.Timeline) (field : Fin 17)
    (floating : (eventCast field).kind=.floating) : DecodedArray Atom :=
  floatingArray ⟨timeline.arrays field,(timeline.kinds field).trans floating⟩

def stringCoordinate (timeline : ERPTemporalNPZ.Timeline) (field : Fin 17)
    (string : (eventCast field).kind=.unicode) : Fin timeline.count → String :=
  let view := timelineStrings timeline field string
  coordinate view.values (view.length.trans (timeline.commonLength field))

def integerCoordinate (timeline : ERPTemporalNPZ.Timeline) (field : Fin 17)
    (integer : (eventCast field).kind=.signed) : Fin timeline.count → ℤ :=
  let view := timelineIntegers timeline field integer
  coordinate view.values (view.length.trans (timeline.commonLength field))

def floatCoordinate (timeline : ERPTemporalNPZ.Timeline) (field : Fin 17)
    (floating : (eventCast field).kind=.floating) : Fin timeline.count → Atom :=
  let view := timelineFloats timeline field floating
  coordinate view.values (view.length.trans (timeline.commonLength field))

def timelineImage (timeline : ERPTemporalNPZ.Timeline) : TimelineImage timeline.count where
  shapes := fun i => (timeline.arrays i).shape
  eventIds := stringCoordinate timeline 0 (by decide)
  groups := stringCoordinate timeline 1 (by decide)
  subjects := stringCoordinate timeline 2 (by decide)
  stimulusDType := .integer
  onsetSampleDType := .integer
  evidenceDType := .integer
  stimulus := integerCoordinate timeline 3 (by decide)
  onsetSamples := integerCoordinate timeline 4 (by decide)
  onset := floatCoordinate timeline 5 (by decide)
  availableAt := floatCoordinate timeline 6 (by decide)
  evidence := integerCoordinate timeline 7 (by decide)
  statuses := stringCoordinate timeline 8 (by decide)
  details := stringCoordinate timeline 9 (by decide)
  datasets := stringCoordinate timeline 10 (by decide)
  sessions := stringCoordinate timeline 11 (by decide)
  runs := stringCoordinate timeline 12 (by decide)
  selections := stringCoordinate timeline 13 (by decide)
  candidates := stringCoordinate timeline 14 (by decide)
  targets := stringCoordinate timeline 15 (by decide)
  repetitions := integerCoordinate timeline 16 (by decide)
  complete := timeline.complete
  onlineCausal := timeline.onlineCausal

end
end N2P3.ERPTemporalArrayViews
