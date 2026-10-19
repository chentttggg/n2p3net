import N2P3.PersonalERPScalar

/-!
# Extended builtin values for the complete recipe-constructor control graph

The153 accepted members are unchanged. This new representation embeds every
old FrozenState.Value and adds IEEE observations, list/tuple distinction and
known NumPy scalar tags. These are decoded observations, not a Python runtime
implementation. Array-valued NumPy objects, custom Python methods, numeric
subclass representation/range, JSON integer digit limits and Unicode strip
implementation have explicit remaining operation obligations.

Only individual int/float casts, finite tests, literal comparisons, string
strip and JSON integer conversion may be backend operations. There is no
FeatureTransferConfig, TargetRecipe, adapter or admission-result primitive.
-/
namespace N2P3.PersonalRecipePython
open N2P3.FrozenState
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

inductive Number where
  | finite (value : ℝ)
  | positiveInfinity | negativeInfinity | nan

mutual
inductive Value where
  | null
  | boolean (value : Bool)
  | integer (value : ℤ)
  | floating (value : Number)
  | numpyInteger (value : ℤ)
  | numpyFloat64 (value : Number)
  | numpyFloat32 (value : Number)
  | numpyBoolean (value : Bool)
  | text (value : String)
  | list (values : Values)
  | tuple (values : Values)
  | object (fields : Fields)
  /-- A builtin dict with at least one nonstring key; **kwargs rejects it before
  the dataclass constructor. Full JSON key normalization is a separate gap. -/
  | nonstringKeyDict
inductive Values where
  | nil
  | cons (head : Value) (tail : Values)
inductive Fields where
  | nil
  | cons (key : String) (value : Value) (tail : Fields)
end

namespace Fields
def ofList : List (String × Value) → Fields
  | [] => .nil
  | (name, value) :: tail => .cons name value (ofList tail)
def get (name : String) : Fields → Option Value
  | .nil => none
  | .cons key value tail => if name = key then some value else get name tail
def keys : Fields → List String
  | .nil => []
  | .cons name _ tail => name :: keys tail
end Fields

mutual
def embed : FrozenState.Value → Value
  | .null => .null
  | .boolean value => .boolean value
  | .integer value => .integer value
  | .number value => .floating (.finite value)
  | .text value => .text value
  | .sequence values => .list (embedValues values)
  | .object fields => .object (embedFields fields)
def embedValues : FrozenState.Values → Values
  | .nil => .nil
  | .cons value tail => .cons (embed value) (embedValues tail)
def embedFields : FrozenState.Fields → Fields
  | .nil => .nil
  | .cons name value tail => .cons name (embed value) (embedFields tail)
end

mutual
def legacy : Value → Option FrozenState.Value
  | .null => some .null
  | .boolean value => some (.boolean value)
  | .integer value => some (.integer value)
  | .floating (.finite value) => some (.number value)
  | .text value => some (.text value)
  | .list values => (legacyValues values).map FrozenState.Value.sequence
  | .object fields => (legacyFields fields).map FrozenState.Value.object
  | _ => none
def legacyValues : Values → Option FrozenState.Values
  | .nil => some .nil
  | .cons value tail => do
    let value ← legacy value
    let tail ← legacyValues tail
    pure (.cons value tail)
def legacyFields : Fields → Option FrozenState.Fields
  | .nil => some .nil
  | .cons name value tail => do
    let value ← legacy value
    let tail ← legacyFields tail
    pure (.cons name value tail)
end

mutual
theorem every_old_value_is_retained : ∀ value : FrozenState.Value, legacy (embed value) = some value
  | .null => rfl
  | .boolean _ => rfl
  | .integer _ => rfl
  | .number _ => rfl
  | .text _ => rfl
  | .sequence values => by simp [embed, legacy, every_old_values_are_retained values]
  | .object fields => by simp [embed, legacy, every_old_fields_are_retained fields]
theorem every_old_values_are_retained : ∀ values : FrozenState.Values, legacyValues (embedValues values) = some values
  | .nil => rfl
  | .cons value tail => by
    simp [embedValues, legacyValues, every_old_value_is_retained value, every_old_values_are_retained tail]
theorem every_old_fields_are_retained : ∀ fields : FrozenState.Fields, legacyFields (embedFields fields) = some fields
  | .nil => rfl
  | .cons name value tail => by
    simp [embedFields, legacyFields, every_old_value_is_retained value, every_old_fields_are_retained tail]
end

mutual
def builtinHashable : Value → Bool
  | .list _ | .object _ | .nonstringKeyDict => false
  | .tuple values => allHashable values
  | _ => true
def allHashable : Values → Bool
  | .nil => true
  | .cons value tail => builtinHashable value && allHashable tail
end

inductive Exception where
  | typeError | valueError | overflowError | attributeError
  | other (name : String)
  | unrefined (operation : String)
  deriving DecidableEq

inductive Relation where
  | lt | le | eq | ne
  deriving DecidableEq

inductive Operation where
  | validation | keywordConstruction | intCast | floatCast | numpyFinite | mathFinite
  | compare (relation : Relation) | setMembership | nameStrip | jsonInteger | jsonDump
  deriving DecidableEq

structure Fault where
  field : String
  operation : Operation
  exception : Exception
  deriving DecidableEq

abbrev Result (α : Type) := Except Fault α

def require (condition : Prop) (field : String) (exception : Exception := .valueError) : Result Unit :=
  if condition then .ok () else .error ⟨field, .validation, exception⟩

def atom (field : String) (operation : Operation) (reply : Except Exception α) : Result α :=
  reply.mapError (fun exception => ⟨field, operation, exception⟩)

structure Primitives where
  intCast : Value → Except Exception ℤ
  floatCast : Value → Except Exception Number
  numpyFinite : Value → Except Exception Bool
  mathFinite : Value → Except Exception Bool
  compare : Relation → Value → Value → Except Exception Bool
  strip : String → Except Exception String
  jsonInteger : ℤ → Except Exception Unit

def isBuiltinBool : Value → Bool
  | .boolean _ => true
  | _ => false
def isIntegral : Value → Bool
  | .boolean _ | .integer _ | .numpyInteger _ => true
  | _ => false
def isReal : Value → Bool
  | .boolean _ | .integer _ | .numpyInteger _ | .floating _ | .numpyFloat64 _ | .numpyFloat32 _ => true
  | _ => false
def recipeScalarTag : Value → Bool
  | .boolean _ | .integer _ | .floating _ | .numpyFloat64 _ => true
  | _ => false
def isStrictInt : Value → Bool
  | .integer _ => true
  | _ => false
def isText (value : Value) (expected : String) : Prop := value = .text expected

def setContains (field : String) (allowed : List (Option String)) (value : Value) : Result Bool :=
  if builtinHashable value then
    pure (match value with
      | .null => allowed.contains none
      | .text text => allowed.contains (some text)
      | _ => false)
  else .error ⟨field, .setMembership, .typeError⟩

def setOption (field : String) (allowed : List (Option String)) (value : Value) : Result Unit := do
  let member ← setContains field allowed value
  require (member = true) field

theorem lists_fail_hashability_before_set_membership (field : String) (allowed : List (Option String)) (values : Values) :
    setOption field allowed (.list values) = .error ⟨field, .setMembership, .typeError⟩ := by
  simp [setOption, setContains, builtinHashable]

theorem empty_tuple_is_hashable_but_not_a_text_option (field : String) (allowed : List (Option String)) :
    setOption field allowed (.tuple .nil) = .error ⟨field, .validation, .valueError⟩ := by
  simp [setOption, setContains, builtinHashable, allHashable, require]

theorem tuple_with_list_is_unhashable (field : String) (allowed : List (Option String)) :
    setOption field allowed (.tuple (.cons (.list .nil) .nil)) = .error ⟨field, .setMembership, .typeError⟩ := by
  simp [setOption, setContains, builtinHashable, allHashable]

theorem nonfinite_float_keeps_numeric_tag :
    recipeScalarTag (.floating .nan) = true ∧ recipeScalarTag (.floating .positiveInfinity) = true := ⟨rfl, rfl⟩

def compareAtom (p : Primitives) (field : String) (relation : Relation) (left right : Value) : Result Bool :=
  atom field (.compare relation) (p.compare relation left right)

def rejectComparison (p : Primitives) (field : String) (relation : Relation) (left right : Value) : Result Unit := do
  let bad ← compareAtom p field relation left right
  require (bad = false) field

def mathBound (p : Primitives) (field : String) (relation : Relation) (left right : Value) : Result Unit := do
  let finite ← atom field .mathFinite (p.mathFinite left)
  require (finite = true) field
  let good ← compareAtom p field relation left right
  require (good = true) field

end
end N2P3.PersonalRecipePython
