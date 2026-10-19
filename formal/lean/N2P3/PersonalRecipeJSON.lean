import N2P3.PersonalTargetRecipe

/-!
# JSON nonfinite rejection and normalization before recipe kwargs construction

Source: calibrated.py:456-457,550 and recipes.py TargetRecipe(**kwargs).
This is an explicit decoded JSON acceptance/normalization specification, not
a proof of Python json.dumps/loads or float decimal conversion. One integer
encoding/digit-limit operation remains a leaf. Builtin tuples normalize to
lists, NumPy float64 to builtin float, unsupported NumPy integers/float32/bool
raise TypeError, and NaN/Inf float observations raise ValueError with
allow_nan=False. The direct constructor's acceptance of an unchecked NaN lr
does not bypass this earlier JSON gate.

Nonstring dictionary-key normalization, Unicode/surrogate and full scalar
codec/IEEE refinements remain explicit obligations. Fields/Values are list
images; arbitrary duplicate-key mathematical cases are not claimed to refine
Python dicts. The generated constructor field images get separate uniqueness
proofs, rather than a whole artifact-validity oracle.
-/
namespace N2P3.PersonalRecipeJSON
open N2P3.PersonalRecipePython
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

mutual
def normalize (p : Primitives) (field : String) : Value → Result Value
  | .null => .ok .null
  | .boolean value => .ok (.boolean value)
  | .integer value => do
    atom field .jsonInteger (p.jsonInteger value)
    pure (.integer value)
  | .floating (.finite value) => .ok (.floating (.finite value))
  | .floating _ => .error ⟨field, .jsonDump, .valueError⟩
  | .numpyFloat64 (.finite value) => .ok (.floating (.finite value))
  | .numpyFloat64 _ => .error ⟨field, .jsonDump, .valueError⟩
  | .numpyInteger _ | .numpyFloat32 _ | .numpyBoolean _ => .error ⟨field, .jsonDump, .typeError⟩
  | .text value => .ok (.text value)
  | .list values | .tuple values => .list <$> normalizeValues p field values
  | .object fields => .object <$> normalizeFields p fields
  | .nonstringKeyDict => .error ⟨field, .jsonDump, .unrefined "JSON nonstring dictionary key normalization"⟩
def normalizeValues (p : Primitives) (field : String) : Values → Result Values
  | .nil => .ok .nil
  | .cons value tail => do
    let value ← normalize p field value
    let tail ← normalizeValues p field tail
    pure (.cons value tail)
def normalizeFields (p : Primitives) : Fields → Result Fields
  | .nil => .ok .nil
  | .cons field value tail => do
    let value ← normalize p field value
    let tail ← normalizeFields p tail
    pure (.cons field value tail)
end

/-- A constructor-composition fragment, not the complete physical calibrated
predictor. The full metadata JSON normalization is before its state clone. -/
def jsonThenRecipe (p : Primitives) (kwargs : Fields) : Result PersonalTargetRecipe.Recipe := do
  let normalized ← normalizeFields p kwargs
  PersonalTargetRecipe.fromKwargs p normalized

theorem builtin_nan_rejected_before_any_recipe (p : Primitives) (tail : Fields) :
    jsonThenRecipe p (.cons "lr" (.floating .nan) tail) =
      .error ⟨"lr", .jsonDump, .valueError⟩ := rfl

theorem positive_infinity_rejected_before_any_recipe (p : Primitives) (tail : Fields) :
    jsonThenRecipe p (.cons "weight_decay" (.floating .positiveInfinity) tail) =
      .error ⟨"weight_decay", .jsonDump, .valueError⟩ := rfl

theorem negative_infinity_rejected_before_any_recipe (p : Primitives) (tail : Fields) :
    jsonThenRecipe p (.cons "seed" (.floating .negativeInfinity) tail) =
      .error ⟨"seed", .jsonDump, .valueError⟩ := rfl

theorem numpy_integer_json_typeerror_precedes_later_nan (p : Primitives) (value : ℤ) :
    normalizeFields p (.cons "epochs" (.numpyInteger value) (.cons "lr" (.floating .nan) .nil)) =
      .error ⟨"epochs", .jsonDump, .typeError⟩ := rfl

theorem numpy_float64_nan_is_json_valueerror (p : Primitives) :
    normalize p "value" (.numpyFloat64 .nan) = .error ⟨"value", .jsonDump, .valueError⟩ := rfl

theorem numpy_float32_nan_is_json_typeerror (p : Primitives) :
    normalize p "value" (.numpyFloat32 .nan) = .error ⟨"value", .jsonDump, .typeError⟩ := rfl

theorem json_tuple_normalizes_to_list (p : Primitives) :
    normalize p "initialization" (.tuple .nil) = .ok (.list .nil) := rfl

theorem direct_empty_tuple_option_and_json_option_have_different_errors
    (p : Primitives) (field : String) (allowed : List (Option String)) :
    setOption field allowed (.tuple .nil) = .error ⟨field, .validation, .valueError⟩ ∧
    ((normalize p field (.tuple .nil)).bind (setOption field allowed)) =
      .error ⟨field, .setMembership, .typeError⟩ := by
  exact ⟨empty_tuple_is_hashable_but_not_a_text_option field allowed,
    lists_fail_hashability_before_set_membership field allowed .nil⟩

theorem json_integer_encoding_exception_is_preserved (p : Primitives) (field : String) (value : ℤ)
    (exception : Exception) (h : p.jsonInteger value = .error exception) :
    normalize p field (.integer value) = .error ⟨field, .jsonInteger, exception⟩ := by
  simp [normalize, atom, h]

end
end N2P3.PersonalRecipeJSON
