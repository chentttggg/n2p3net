import N2P3.ERPTemporalMultiScope

/-!
# json.loads values before source finite/type guards

Default CPython json.loads accepts NaN/Infinity and uses its shared decoder
constants for those spellings. Direct NaN numeric comparison is false, while
list/dict element identity can make two occurrences of the SAME decoder NaN
compare equal. This layer represents those decoded trees, including values
that the finite FrozenState.Value boundary cannot represent. It does not
reject them early or assert that finite Value/PyEq handles arbitrary JSON.

Evidence: p12_json_nonfinite_plan_01.json/probe_01.json fixed and run before
this module; stdlib-only CPython 3.14.3, no source fit or production import.
Canonical object fields below describe unique-key decoded dictionaries.
JSON parsing/printing and Python codec implementation remain local boundaries.
-/

namespace N2P3.ERPTemporalJSON

open N2P3.ERPTemporalMultiScope

noncomputable section

mutual
inductive JValue where
  | null
  | boolean (value : Bool)
  | integer (value : ℤ)
  | floating (value : FloatAtom)
  | negativeZero
  | text (value : String)
  | sequence (values : JValues)
  | object (fields : JFields)
inductive JValues where
  | nil
  | cons (value : JValue) (tail : JValues)
inductive JFields where
  | nil
  | cons (key : String) (value : JValue) (tail : JFields)
end

inductive Canonical where
  | null
  | numeric (value : ℝ)
  | nan | positiveInfinity | negativeInfinity
  | text (value : String)
  | sequence (values : List Canonical)
  | object (fields : List (String × Canonical))

mutual
def normalize : JValue → Canonical
  | .null => .null
  | .boolean value => .numeric (if value then 1 else 0)
  | .integer value => .numeric value
  | .floating (.finite value) => .numeric value
  | .negativeZero => .numeric 0
  | .floating .nan => .nan
  | .floating .positiveInfinity => .positiveInfinity
  | .floating .negativeInfinity => .negativeInfinity
  | .text value => .text value
  | .sequence values => .sequence (normalizeValues values)
  | .object fields => .object ((normalizeFields fields).mergeSort (fun a b => a.1 ≤ b.1))
def normalizeValues : JValues → List Canonical
  | .nil => []
  | .cons value tail => normalize value :: normalizeValues tail
def normalizeFields : JFields → List (String × Canonical)
  | .nil => []
  | .cons key value tail => (key,normalize value) :: normalizeFields tail
end

def scalarNaN : JValue → Prop
  | .floating .nan => True
  | _ => False

/-- Only decoded JSON trees with the shared CPython NaN constants. Object
comparison ignores key order; numeric tags follow Python equality. -/
def pythonEqual (a b : JValue) : Prop :=
  ¬scalarNaN a ∧ ¬scalarNaN b ∧ normalize a=normalize b

def JFields.get (key : String) : JFields → Option JValue
  | .nil => none
  | .cons name value tail => if name=key then some value else tail.get key

def JFields.ofList : List (String × JValue) → JFields
  | [] => .nil
  | (key,value) :: tail => .cons key value (JFields.ofList tail)

def JFields.keys : JFields → List String
  | .nil => []
  | .cons key _ tail => key :: tail.keys

def JValues.ofList : List JValue → JValues
  | [] => .nil
  | value :: tail => .cons value (JValues.ofList tail)

def JValues.toList : JValues → List JValue
  | .nil => []
  | .cons value tail => value :: tail.toList

mutual
def fromValue : N2P3.FrozenState.Value → JValue
  | .null => .null
  | .boolean value => .boolean value
  | .integer value => .integer value
  | .number value => .floating (.finite value)
  | .text value => .text value
  | .sequence values => .sequence (fromValues values)
  | .object fields => .object (fromFields fields)
def fromValues : N2P3.FrozenState.Values → JValues
  | .nil => .nil
  | .cons value tail => .cons (fromValue value) (fromValues tail)
def fromFields : N2P3.FrozenState.Fields → JFields
  | .nil => .nil
  | .cons key value tail => .cons key (fromValue value) (fromFields tail)
end

mutual
/-- A partial mathematical projection, NOT a loader rejection gate. Keep the
raw value when this returns none until the source actually checks it. -/
def finiteView : JValue → Option N2P3.FrozenState.Value
  | .null => some .null
  | .boolean value => some (.boolean value)
  | .integer value => some (.integer value)
  | .floating (.finite value) => some (.number value)
  | .negativeZero => some (.number 0)
  | .floating _ => none
  | .text value => some (.text value)
  | .sequence values => (.sequence ·) <$> finiteValues values
  | .object fields => (.object ·) <$> finiteFields fields
def finiteValues : JValues → Option N2P3.FrozenState.Values
  | .nil => some .nil
  | .cons value tail => do
    let value ← finiteView value
    let tail ← finiteValues tail
    pure (.cons value tail)
def finiteFields : JFields → Option N2P3.FrozenState.Fields
  | .nil => some .nil
  | .cons key value tail => do
    let value ← finiteView value
    let tail ← finiteFields tail
    pure (.cons key value tail)
end

theorem scalar_decoded_nan_is_not_reflexive : ¬ pythonEqual (.floating .nan) (.floating .nan) := by
  simp [pythonEqual,scalarNaN]

theorem list_elements_share_decoder_nan_identity :
    pythonEqual (.sequence (.cons (.floating .nan) .nil)) (.sequence (.cons (.floating .nan) .nil)) := by
  simp [pythonEqual,scalarNaN]

theorem nested_dictionary_decoder_nan_identity :
    pythonEqual (.object (.cons "sfreq" (.floating .nan) .nil))
      (.object (.cons "sfreq" (.floating .nan) .nil)) := by
  simp [pythonEqual,scalarNaN]

theorem nonfinite_raw_json_has_no_finite_view :
    finiteView (.floating .nan)=none ∧ finiteView (.floating .positiveInfinity)=none ∧
      finiteView (.floating .negativeInfinity)=none := ⟨rfl,rfl,rfl⟩

end
end N2P3.ERPTemporalJSON
