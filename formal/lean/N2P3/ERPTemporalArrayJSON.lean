import N2P3.ERPTemporalArrayViews
import N2P3.ERPTemporalJSON

/-! Shape-preserving ndarray.tolist projections and the actual Python set
operation used by materialize_dataset_identity. Scalar Unicode arrays yield
a string, so set() iterates its characters; multidimensional arrays yield
nested lists and set() raises for unhashable elements. Neither case is
replaced by an early subject-vector validation.
-/

namespace N2P3.ERPTemporalArrayJSON

open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalERPData
open N2P3.ERPTemporalNPZ N2P3.ERPTemporalArrayViews N2P3.ERPTemporalJSON

noncomputable section

def nest (shape : List ℕ) (values : List JValue) : JValue :=
  match shape with
  | [] => values.headD .null
  | first :: tail => .sequence (JValues.ofList ((List.range first).map (fun index =>
      nest tail ((values.drop (index*tail.prod)).take tail.prod))))

def toList {α : Type} (view : DecodedArray α) (encode : α → JValue) : JValue :=
  nest view.raw.shape (view.values.map encode)

def strings (image : {raw : RawArray // raw.kind=.unicode}) : JValue :=
  toList (stringArray image) JValue.text

def booleans (image : {raw : RawArray // raw.kind=.boolean}) : JValue :=
  toList (booleanArray image) JValue.boolean

def hashableSubjectElements : List JValue → Check (List String)
  | [] => .ok []
  | .text string :: tail => do
    let rest ← hashableSubjectElements tail
    pure (string :: rest)
  | _ :: _ => .error (.fieldFailure "unhashable list in subject-id set")

def pythonSubjectSet : JValue → Check (List String)
  | .text value => .ok ((value.toList.map (fun char => String.singleton char)).dedup)
  | .sequence values => (hashableSubjectElements values.toList).map List.dedup
  | _ => .error (.fieldFailure "subject-id set() iteration")

theorem scalar_subject_set_iterates_characters :
    pythonSubjectSet (.text "AB")=.ok ["A","B"] := rfl

theorem nested_subject_lists_are_not_flattened :
    pythonSubjectSet (.sequence (.cons (.sequence (.cons (.text "A") .nil)) .nil))=
      .error (.fieldFailure "unhashable list in subject-id set") := rfl

end
end N2P3.ERPTemporalArrayJSON
