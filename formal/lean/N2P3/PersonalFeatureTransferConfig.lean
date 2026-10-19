import N2P3.PersonalRecipePython

/-!
# FeatureTransferConfig kwargs, conversions and ordered validation

Source: models/adapters.py:160-228; recipes.py:41-53. Six Integral checks/int
conversions precede three Real checks/float conversions. All conversions
precede range checks. This is the ordinary kwargs dataclass constructor, not
feature_transfer_config_from_record (whose schema/readback rules differ).

No whole configuration-success callback is used. Raw integer/float tags and
each cast/finite/comparison response are separate. Direct NaN/Inf and tuple
values are represented and reach their source checks. Underlying NumPy scalar
class/range representation and Python/IEEE implementations remain independent
leaf/refinement obligations; this module does not declare complete P01.
-/
namespace N2P3.PersonalFeatureTransferConfig
open N2P3.PersonalRecipePython
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def quarter : Number := .finite (1 / 4)
def twoTenths : Number := .finite (3602879701896397 / 18014398509481984)
def threeTenths : Number := .finite (5404319552844595 / 18014398509481984)
def half : Number := .finite (1 / 2)
def zero : Value := .floating (.finite 0)

structure Raw where
  invariant : Value := .integer 4
  privateChannels : Value := .integer 4
  target : Value := .integer 4
  kernel : Value := .integer 9
  maxInvariant : Value := .floating quarter
  maxTarget : Value := .floating quarter
  privateCount : Value := .integer 0
  sourceCount : Value := .integer 0
  drop : Value := .floating threeTenths
  dropMode : Value := .text "sample"

def kwargsFields (r : Raw) : Fields := Fields.ofList
  [("invariant_bottleneck_channels", r.invariant), ("private_channels", r.privateChannels),
    ("target_bottleneck_channels", r.target), ("target_kernel_size", r.kernel),
    ("max_invariant_residual", r.maxInvariant), ("max_target_residual", r.maxTarget),
    ("private_domain_count", r.privateCount), ("source_domain_count", r.sourceCount),
    ("source_drop_p", r.drop), ("source_drop_mode", r.dropMode)]

def names : List String := Fields.keys (kwargsFields {})
def getDefault (fields : Fields) (name : String) (default : Value) : Value := (fields.get name).getD default

def readArgs (fields : Fields) : Result Raw := do
  if ¬fields.keys ⊆ names then
    .error ⟨"FeatureTransferConfig kwargs", .keywordConstruction, .typeError⟩
  else pure {
    invariant := getDefault fields "invariant_bottleneck_channels" (.integer 4)
    privateChannels := getDefault fields "private_channels" (.integer 4)
    target := getDefault fields "target_bottleneck_channels" (.integer 4)
    kernel := getDefault fields "target_kernel_size" (.integer 9)
    maxInvariant := getDefault fields "max_invariant_residual" (.floating quarter)
    maxTarget := getDefault fields "max_target_residual" (.floating quarter)
    privateCount := getDefault fields "private_domain_count" (.integer 0)
    sourceCount := getDefault fields "source_domain_count" (.integer 0)
    drop := getDefault fields "source_drop_p" (.floating threeTenths)
    dropMode := getDefault fields "source_drop_mode" (.text "sample") }

structure Config where
  invariant : ℤ
  privateChannels : ℤ
  target : ℤ
  kernel : ℤ
  maxInvariant : Number
  maxTarget : Number
  privateCount : ℤ
  sourceCount : ℤ
  drop : Number
  dropMode : Value

def asRaw (c : Config) : Raw :=
  ⟨.integer c.invariant, .integer c.privateChannels, .integer c.target, .integer c.kernel,
    .floating c.maxInvariant, .floating c.maxTarget, .integer c.privateCount,
    .integer c.sourceCount, .floating c.drop, c.dropMode⟩

def readIntegral (p : Primitives) (field : String) (value : Value) : Result ℤ := do
  require (isBuiltinBool value = false ∧ isIntegral value = true) field .typeError
  atom field .intCast (p.intCast value)

def readReal (p : Primitives) (field : String) (value : Value) : Result Number := do
  require (isBuiltinBool value = false ∧ isReal value = true) field .typeError
  atom field .floatCast (p.floatCast value)

def finitePositive (p : Primitives) (field : String) (number : Number) : Result Unit := do
  let finite ← atom field .mathFinite (p.mathFinite (.floating number))
  require (finite = true) field
  rejectComparison p field .le (.floating number) zero

def dropBound (p : Primitives) (number : Number) : Result Unit := do
  let finite ← atom "source_drop_p" .mathFinite (p.mathFinite (.floating number))
  require (finite = true) "source_drop_p"
  let isZero ← compareAtom p "source_drop_p zero" .eq (.floating number) zero
  if isZero then pure ()
  else
    let low ← compareAtom p "source_drop_p lower" .le (.floating twoTenths) (.floating number)
    require (low = true) "source_drop_p range"
    let high ← compareAtom p "source_drop_p upper" .le (.floating number) (.floating half)
    require (high = true) "source_drop_p range"

def construct (p : Primitives) (r : Raw) : Result Config := do
  let invariant ← readIntegral p "invariant_bottleneck_channels" r.invariant
  let privateChannels ← readIntegral p "private_channels" r.privateChannels
  let target ← readIntegral p "target_bottleneck_channels" r.target
  let kernel ← readIntegral p "target_kernel_size" r.kernel
  let privateCount ← readIntegral p "private_domain_count" r.privateCount
  let sourceCount ← readIntegral p "source_domain_count" r.sourceCount
  let maxInvariant ← readReal p "max_invariant_residual" r.maxInvariant
  let maxTarget ← readReal p "max_target_residual" r.maxTarget
  let drop ← readReal p "source_drop_p" r.drop
  require (1 ≤ invariant ∧ 1 ≤ privateChannels ∧ 1 ≤ target) "feature-transfer channel counts"
  require (3 ≤ kernel ∧ kernel % 2 ≠ 0) "target_kernel_size"
  finitePositive p "max_invariant_residual" maxInvariant
  finitePositive p "max_target_residual" maxTarget
  require (privateCount ≠ 1 ∧ 0 ≤ privateCount) "private_domain_count"
  require (sourceCount ≠ 1 ∧ 0 ≤ sourceCount) "source_domain_count"
  dropBound p drop
  setOption "source_drop_mode" [some "batch", some "sample"] r.dropMode
  require (¬(0 < privateCount ∧ 0 < sourceCount ∧ privateCount ≠ sourceCount)) "domain-count agreement"
  pure ⟨invariant, privateChannels, target, kernel, maxInvariant, maxTarget,
    privateCount, sourceCount, drop, r.dropMode⟩

def fromKwargs (p : Primitives) (fields : Fields) : Result Config := do
  let raw ← readArgs fields
  construct p raw

theorem explicit_kwargs_reconstruct_each_config_field (r : Raw) : readArgs (kwargsFields r) = .ok r := by
  cases r
  simp [readArgs, kwargsFields, names, Fields.ofList, Fields.keys, Fields.get, getDefault]

theorem unknown_config_keyword_precedes_every_numeric_operation (p : Primitives) (fields : Fields)
    (h : ¬fields.keys ⊆ names) :
    fromKwargs p fields = .error ⟨"FeatureTransferConfig kwargs", .keywordConstruction, .typeError⟩ := by
  simp [fromKwargs, readArgs, h]

theorem bool_is_not_feature_integral (p : Primitives) (field : String) (value : Bool) :
    readIntegral p field (.boolean value) = .error ⟨field, .validation, .typeError⟩ := by
  simp [readIntegral, isBuiltinBool, isIntegral, require]

theorem finite_float_is_not_feature_integral (p : Primitives) (field : String) (value : ℝ) :
    readIntegral p field (.floating (.finite value)) = .error ⟨field, .validation, .typeError⟩ := by
  simp [readIntegral, isBuiltinBool, isIntegral, require]

theorem numpy_integer_reaches_its_individual_cast (p : Primitives) (field : String) (value : ℤ) :
    readIntegral p field (.numpyInteger value) = atom field .intCast (p.intCast (.numpyInteger value)) := by
  simp [readIntegral, isBuiltinBool, isIntegral, require]

theorem first_integral_typeerror_precedes_all_conversions (p : Primitives) (r : Raw) (value : Bool)
    (h : r.invariant = .boolean value) :
    construct p r = .error ⟨"invariant_bottleneck_channels", .validation, .typeError⟩ := by
  simp [construct, h, bool_is_not_feature_integral]

theorem nonfinite_real_tag_does_not_skip_float_conversion (p : Primitives) (field : String) :
    readReal p field (.floating .nan) = atom field .floatCast (p.floatCast (.floating .nan)) := by
  simp [readReal, isBuiltinBool, isReal, require]

structure IntegerFields (p : Primitives) (r : Raw) (c : Config) : Prop where
  invariantTag : isBuiltinBool r.invariant = false ∧ isIntegral r.invariant = true
  privateTag : isBuiltinBool r.privateChannels = false ∧ isIntegral r.privateChannels = true
  targetTag : isBuiltinBool r.target = false ∧ isIntegral r.target = true
  kernelTag : isBuiltinBool r.kernel = false ∧ isIntegral r.kernel = true
  privateCountTag : isBuiltinBool r.privateCount = false ∧ isIntegral r.privateCount = true
  sourceCountTag : isBuiltinBool r.sourceCount = false ∧ isIntegral r.sourceCount = true
  invariantCast : p.intCast r.invariant = .ok c.invariant
  privateCast : p.intCast r.privateChannels = .ok c.privateChannels
  targetCast : p.intCast r.target = .ok c.target
  kernelCast : p.intCast r.kernel = .ok c.kernel
  privateCountCast : p.intCast r.privateCount = .ok c.privateCount
  sourceCountCast : p.intCast r.sourceCount = .ok c.sourceCount

structure CastFields (p : Primitives) (r : Raw) (c : Config) : Prop extends IntegerFields p r c where
  invariantRealTag : isBuiltinBool r.maxInvariant = false ∧ isReal r.maxInvariant = true
  targetRealTag : isBuiltinBool r.maxTarget = false ∧ isReal r.maxTarget = true
  dropRealTag : isBuiltinBool r.drop = false ∧ isReal r.drop = true
  invariantFloat : p.floatCast r.maxInvariant = .ok c.maxInvariant
  targetFloat : p.floatCast r.maxTarget = .ok c.maxTarget
  dropFloat : p.floatCast r.drop = .ok c.drop
  mode : r.dropMode = c.dropMode

structure GuardFields (p : Primitives) (c : Config) : Prop where
  channels : 1 ≤ c.invariant ∧ 1 ≤ c.privateChannels ∧ 1 ≤ c.target
  kernel : 3 ≤ c.kernel ∧ c.kernel % 2 ≠ 0
  invariantFinite : p.mathFinite (.floating c.maxInvariant) = .ok true
  invariantBound : p.compare .le (.floating c.maxInvariant) zero = .ok false
  targetFinite : p.mathFinite (.floating c.maxTarget) = .ok true
  targetBound : p.compare .le (.floating c.maxTarget) zero = .ok false
  privateCount : c.privateCount ≠ 1 ∧ 0 ≤ c.privateCount
  sourceCount : c.sourceCount ≠ 1 ∧ 0 ≤ c.sourceCount
  dropFinite : p.mathFinite (.floating c.drop) = .ok true
  dropZeroOrRange :
    p.compare .eq (.floating c.drop) zero = .ok true ∨
    (p.compare .eq (.floating c.drop) zero = .ok false ∧
      p.compare .le (.floating twoTenths) (.floating c.drop) = .ok true ∧
      p.compare .le (.floating c.drop) (.floating half) = .ok true)
  dropMode : c.dropMode = .text "batch" ∨ c.dropMode = .text "sample"
  agreement : ¬(0 < c.privateCount ∧ 0 < c.sourceCount ∧ c.privateCount ≠ c.sourceCount)

theorem config_from_raw_fields_and_individual_operations (p : Primitives) (r : Raw) (c : Config)
    (a : CastFields p r c) (g : GuardFields p c) : construct p r = .ok c := by
  have hm : setOption "source_drop_mode" [some "batch", some "sample"] r.dropMode = .ok () := by
    rw [a.mode]
    rcases g.dropMode with hm | hm <;> simp [setOption, setContains, hm, builtinHashable, require]
  have hd : dropBound p c.drop = .ok () := by
    rcases g.dropZeroOrRange with hz | ⟨hz, hl, hh⟩
    · simp [dropBound, atom, compareAtom, g.dropFinite, hz, require]
    · simp [dropBound, atom, compareAtom, g.dropFinite, hz, hl, hh, require]
  have hc : Config.mk c.invariant c.privateChannels c.target c.kernel c.maxInvariant c.maxTarget
      c.privateCount c.sourceCount c.drop r.dropMode = c := by
    rw [a.mode]
  simpa only [hc] using (show construct p r = .ok (Config.mk c.invariant c.privateChannels c.target c.kernel
      c.maxInvariant c.maxTarget c.privateCount c.sourceCount c.drop r.dropMode) by
    simp [construct, readIntegral, readReal, require, atom,
      a.invariantTag, a.privateTag, a.targetTag, a.kernelTag, a.privateCountTag, a.sourceCountTag,
      a.invariantCast, a.privateCast, a.targetCast, a.kernelCast, a.privateCountCast, a.sourceCountCast,
      a.invariantRealTag, a.targetRealTag, a.dropRealTag, a.invariantFloat, a.targetFloat, a.dropFloat,
      g.channels, g.kernel, finitePositive, rejectComparison, compareAtom,
      g.invariantFinite, g.invariantBound, g.targetFinite, g.targetBound,
      g.privateCount, g.sourceCount, hd, hm, g.agreement])

theorem kwargs_config_from_independent_fields (p : Primitives) (r : Raw) (c : Config)
    (a : CastFields p r c) (g : GuardFields p c) : fromKwargs p (kwargsFields r) = .ok c := by
  rw [fromKwargs, explicit_kwargs_reconstruct_each_config_field]
  exact config_from_raw_fields_and_individual_operations p r c a g

theorem first_float_exception_precedes_all_range_checks (p : Primitives) (r : Raw) (c : Config)
    (a : IntegerFields p r c)
    (tag : isBuiltinBool r.maxInvariant = false ∧ isReal r.maxInvariant = true)
    (exception : Exception) (he : p.floatCast r.maxInvariant = .error exception) :
    construct p r = .error ⟨"max_invariant_residual", .floatCast, exception⟩ := by
  simp [construct, readIntegral, readReal, require, atom, tag, he,
    a.invariantTag, a.privateTag, a.targetTag, a.kernelTag, a.privateCountTag, a.sourceCountTag,
    a.invariantCast, a.privateCast, a.targetCast, a.kernelCast, a.privateCountCast, a.sourceCountCast]

theorem final_real_typeerror_precedes_domain_count_ranges (p : Primitives) (r : Raw) (c : Config)
    (a : IntegerFields p r c)
    (ti : isBuiltinBool r.maxInvariant = false ∧ isReal r.maxInvariant = true)
    (tt : isBuiltinBool r.maxTarget = false ∧ isReal r.maxTarget = true)
    (hi : p.floatCast r.maxInvariant = .ok c.maxInvariant)
    (ht : p.floatCast r.maxTarget = .ok c.maxTarget)
    (bad : ¬(isBuiltinBool r.drop = false ∧ isReal r.drop = true)) :
    construct p r = .error ⟨"source_drop_p", .validation, .typeError⟩ := by
  simp [construct, readIntegral, readReal, require, atom, ti, tt, hi, ht, bad,
    a.invariantTag, a.privateTag, a.targetTag, a.kernelTag, a.privateCountTag, a.sourceCountTag,
    a.invariantCast, a.privateCast, a.targetCast, a.kernelCast, a.privateCountCast, a.sourceCountCast]

theorem kernel_rejection_follows_all_conversions (p : Primitives) (r : Raw) (c : Config)
    (a : CastFields p r c) (hc : 1 ≤ c.invariant ∧ 1 ≤ c.privateChannels ∧ 1 ≤ c.target)
    (hk : ¬(3 ≤ c.kernel ∧ c.kernel % 2 ≠ 0)) :
    construct p r = .error ⟨"target_kernel_size", .validation, .valueError⟩ := by
  simp [construct, readIntegral, readReal, require, atom, hc, hk,
    a.invariantTag, a.privateTag, a.targetTag, a.kernelTag, a.privateCountTag, a.sourceCountTag,
    a.invariantCast, a.privateCast, a.targetCast, a.kernelCast, a.privateCountCast, a.sourceCountCast,
    a.invariantRealTag, a.targetRealTag, a.dropRealTag, a.invariantFloat, a.targetFloat, a.dropFloat]

end
end N2P3.PersonalFeatureTransferConfig
