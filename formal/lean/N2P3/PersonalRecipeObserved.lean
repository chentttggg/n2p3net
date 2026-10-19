import N2P3.PersonalTargetRecipe

/-!
# Declared individual-operation witnesses for the fixed constructor controls

The source plan was fixed before any of the35 constructor calls. This module
supplies single numeric cast/finite/comparison/name operations for those
decoded witnesses, never a config/recipe result. It is not a universal NumPy,
math, Python float or JSON implementation. Unobserved integer representability,
string-strip and JSON digit-limit cases are explicit leaf obligations.
-/
namespace N2P3.PersonalRecipeObserved
open N2P3.PersonalRecipePython
noncomputable section
set_option exponentiation.threshold 1024
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def huge100 : ℤ := 10 ^ 100
def huge400 : ℤ := 10 ^ 400

def numberOf : Value → Option Number
  | .integer value | .numpyInteger value => some (.finite value)
  | .floating value | .numpyFloat64 value | .numpyFloat32 value => some value
  | .boolean value | .numpyBoolean value => some (.finite (if value then 1 else 0))
  | _ => none

def numberFinite : Number → Bool
  | .finite _ => true
  | _ => false

def numberLt : Number → Number → Prop
  | .nan, _ | _, .nan => False
  | .finite left, .finite right => left < right
  | .negativeInfinity, .finite _ | .negativeInfinity, .positiveInfinity | .finite _, .positiveInfinity => True
  | _, _ => False

def numberLe : Number → Number → Prop
  | .nan, _ | _, .nan => False
  | .finite left, .finite right => left ≤ right
  | .negativeInfinity, _ | _, .positiveInfinity => True
  | _, _ => False

def numberEq : Number → Number → Prop
  | .finite left, .finite right => left = right
  | .positiveInfinity, .positiveInfinity | .negativeInfinity, .negativeInfinity => True
  | _, _ => False

def compareNumbers (relation : Relation) (left right : Number) : Bool :=
  decide (match relation with
    | .lt => numberLt left right
    | .le => numberLe left right
    | .eq => numberEq left right
    | .ne => ¬numberEq left right)

def ordinaryCompare (relation : Relation) (left right : Value) : Except Exception Bool :=
  match numberOf left, numberOf right with
  | some left, some right => .ok (compareNumbers relation left right)
  | _, _ => match relation with
    | .lt | .le => .error .typeError
    | _ => .error (.unrefined "nonnumeric equality outside literal-numeric constructor operations")

def intCast : Value → Except Exception ℤ
  | .integer value | .numpyInteger value => .ok value
  | _ => .error (.unrefined "int cast outside the checked Integral tags")

def floatInteger (value : ℤ) : Except Exception Number :=
  if value = huge400 then .error .overflowError
  else if -64 ≤ value ∧ value ≤ 64 then .ok (.finite value)
  else .error (.unrefined "unobserved int to float64 conversion")

def floatCast : Value → Except Exception Number
  | .integer value | .numpyInteger value => floatInteger value
  | .floating value | .numpyFloat64 value | .numpyFloat32 value => .ok value
  | _ => .error (.unrefined "float cast outside checked Real tags")

def numpyFinite : Value → Except Exception Bool
  | .integer value =>
    if value = huge100 ∨ value = huge400 then .error .typeError
    else if -64 ≤ value ∧ value ≤ 64 then .ok true
    else .error (.unrefined "unobserved Python integer numpy.isfinite representation")
  | .numpyInteger _ | .boolean _ | .numpyBoolean _ => .ok true
  | .floating value | .numpyFloat64 value | .numpyFloat32 value => .ok (numberFinite value)
  | _ => .error (.unrefined "nonscalar numpy.isfinite output outside scalar type gate")

def mathFinite : Value → Except Exception Bool
  | .integer value =>
    if value = huge400 then .error .overflowError
    else if value = huge100 ∨ (-64 ≤ value ∧ value ≤ 64) then .ok true
    else .error (.unrefined "unobserved Python integer math.isfinite conversion")
  | .numpyInteger _ | .boolean _ | .numpyBoolean _ => .ok true
  | .floating value | .numpyFloat64 value | .numpyFloat32 value => .ok (numberFinite value)
  | _ => .error .typeError

def stripObserved (text : String) : Except Exception String :=
  if text ∈ ["full_recipe_probe", "source", "scratch", "waveform_lda", "source_feature_lda",
      "pooled_lda", "toeplitz_lda", "temporal_lda", "xdawn_erp", ""] then .ok text
  else .error (.unrefined "unobserved Unicode string.strip input")

def jsonInteger (value : ℤ) : Except Exception Unit :=
  if -huge100 ≤ value ∧ value ≤ huge100 then .ok ()
  else .error (.unrefined "JSON integer digit limits and conversion")

def primitives : Primitives :=
  ⟨intCast, floatCast, numpyFinite, mathFinite, ordinaryCompare, stripObserved, jsonInteger⟩

theorem huge_numpy_typeerror : primitives.numpyFinite (.integer huge100) = .error .typeError := by
  simp [primitives, numpyFinite]

theorem huge_float_cast_overflow : primitives.floatCast (.integer huge400) = .error .overflowError := by
  simp [primitives, floatCast, floatInteger]

theorem direct_nan_optimizer_comparisons_are_false :
    primitives.compare .lt (.floating .nan) (.integer 1) = .ok false ∧
    primitives.compare .le (.floating .nan) (.floating (.finite 0)) = .ok false := by
  simp [primitives, ordinaryCompare, numberOf, compareNumbers, numberLt, numberLe]

theorem nan_finite_tests_return_false :
    primitives.numpyFinite (.floating .nan) = .ok false ∧
    primitives.mathFinite (.floating .nan) = .ok false := ⟨rfl, rfl⟩

theorem builtin_bool_finite_test_does_not_remove_the_separate_type_guard :
    primitives.numpyFinite (.boolean true) = .ok true ∧
    PersonalTargetRecipe.scalarField primitives "candidate_loss_temperature" false (.boolean true) =
      .error ⟨"candidate_loss_temperature", .validation, .valueError⟩ := by
  simp [primitives, numpyFinite, PersonalTargetRecipe.scalarField, isBuiltinBool, require]

end
end N2P3.PersonalRecipeObserved
