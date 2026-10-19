import N2P3.PersonalERPScalar

/-!
# Independent raw fields and single scalar operations for plain ERP recipes

The success theorem takes raw field equalities and one response for each
actual NumPy/math/comparison primitive. It does not assume recipe-constructor
success. Arbitrary Value scalars, including unbounded integers and containers,
remain in the executable Scalar consumer; failures are not excluded by this
conditional success theorem. Plain here excludes temporal declarations only;
the separate consumer slice further excludes xDAWN numeric-state construction.
-/
namespace N2P3.PersonalERPScalarFields
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.ERPState
open N2P3.PersonalERPRecipe N2P3.PersonalERPScalar
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

structure PlainStructure (r : RawRecipe) (family : Family) (width : ℕ) (name : String) : Prop where
  initialization : r.initialization = .text family.text
  nameField : r.name = .text name
  nonblank : Nonblank name
  inputStatistics : r.inputStatistics = .text "source" ∨ r.inputStatistics = .text "target_prefix"
  buffers : r.bnBuffers = .text "source" ∨ r.bnBuffers = .text "calibration"
  affine : r.bnAffine = .text "frozen"
  weights : r.weights = .text "none"
  dropout : r.dropout = .text "off"
  adapter : r.targetAdapterConfig = .null
  loss : r.candidateLossWeight = .null
  widthField : r.erpBinWidth = .integer width
  widthPositive : 0 < width
  plain : family ≠ .temporal
  noKind : r.temporalBasisKind = .null
  noRank : r.temporalRank = .null
  maxInteger : r.maxUpdates = .null ∨ ExactInt r.maxUpdates
  refitBoolean : ExactBool r.refitFullPrefix

structure ScalarAtoms (s : Primitives) (r : RawRecipe) : Prop where
  temperatureTag : numericNonBool r.candidateLossTemperature = true
  priorTag : numericNonBool r.sourcePriorWeight = true
  numpyTemperature : s.numpyIsFinite r.candidateLossTemperature = .ok true
  numpyPrior : s.numpyIsFinite r.sourcePriorWeight = .ok true
  temperatureNegative : s.compare .ltZero r.candidateLossTemperature = .ok false
  priorNegative : s.compare .ltZero r.sourcePriorWeight = .ok false
  temperatureNonpositive : s.compare .leZero r.candidateLossTemperature = .ok false
  priorNonzero : s.compare .neZero r.sourcePriorWeight = .ok false
  mathPrior : s.mathIsFinite r.sourcePriorWeight = .ok true
  priorNonnegative : s.compare .zeroLe r.sourcePriorWeight = .ok true
  epochBound : s.compare .ltOne r.epochs = .ok false
  batchBound : s.compare .ltOne r.batchSize = .ok false
  learningRateBound : s.compare .leZero r.lr = .ok false
  updateBound : r.maxUpdates ≠ .null → s.compare .ltOne r.maxUpdates = .ok false
  mathDefaultLoss : s.mathIsFinite (.number 0) = .ok true
  defaultLossNonnegative : s.compare .zeroLe (.number 0) = .ok true
  mathTemperature : s.mathIsFinite r.candidateLossTemperature = .ok true
  temperaturePositive : s.compare .zeroLt r.candidateLossTemperature = .ok true

def plainRecipe (r : RawRecipe) (family : Family) (width : ℕ) : Recipe :=
  ⟨r, family, width, none, none, adapterProjection r⟩

theorem scalar_prefix_from_individual_atoms (s : Primitives) (r : RawRecipe)
    (hloss : r.candidateLossWeight = .null) (h : ScalarAtoms s r) :
    (scalarPrefix s r).result = .ok () := by
  have ht := scalar_field_succeeds s "candidate_loss_temperature" false r.candidateLossTemperature
    h.temperatureTag h.numpyTemperature h.temperatureNegative
  have hp := scalar_field_succeeds s "source_prior_weight" false r.sourcePriorWeight
    h.priorTag h.numpyPrior h.priorNegative
  simp [scalarPrefix, hloss, loss_none_skips_every_operation, ht, hp, andThen,
    compareCall, atomicCall, h.temperatureNonpositive, done]

theorem plain_option_fields_derive_membership (r : RawRecipe) (family : Family) (width : ℕ)
    (name : String) (h : PlainStructure r family width name) : recipeOptions r = .ok () := by
  rcases h.inputStatistics with hs | hs <;> rcases h.buffers with hb | hb <;>
    cases family <;> simp [recipeOptions, setOption, setContains, builtinHashable,
      require, h.initialization, hs, hb, h.affine, h.weights, h.dropout,
      Family.text, PersonalAdmission.PyEq, PersonalAdmission.normalizeValue]

theorem plain_adapter_checks_from_individual_atoms (s : Primitives) (r : RawRecipe)
    (family : Family) (width : ℕ) (name : String) (h : PlainStructure r family width name)
    (a : ScalarAtoms s r) : adapterChecks s r = .ok () := by
  have hm : r.maxUpdates = .null ∨
      (ExactInt r.maxUpdates ∧ s.compare .ltOne r.maxUpdates = .ok false) := by
    rcases h.maxInteger with hn | hi
    · exact Or.inl hn
    · by_cases hn : r.maxUpdates = .null
      · exact Or.inl hn
      · exact Or.inr ⟨hi, a.updateBound hn⟩
  rcases h.inputStatistics with hs | hs <;> rcases h.buffers with hb | hb <;>
    rcases hm with hm | ⟨hi, hm⟩ <;>
    norm_num [adapterChecks, adapterProjection, h.weights, h.affine, h.dropout, h.loss,
      hs, hb, hm, h.refitBoolean, optimization, mathBound, finiteResult, rejectWhen,
      compareResult, compareCall, atomicCall, a.mathPrior, a.priorNonnegative,
      a.epochBound, a.batchBound, a.learningRateBound, a.mathDefaultLoss,
      a.defaultLossNonnegative, a.mathTemperature, a.temperaturePositive,
      setOption, setContains, builtinHashable, require, PersonalAdmission.PyEq,
      PersonalAdmission.normalizeValue]
  all_goals simp_all

theorem plain_constructor_from_raw_fields_and_atoms (s : Primitives) (r : RawRecipe)
    (family : Family) (width : ℕ) (name : String) (h : PlainStructure r family width name)
    (a : ScalarAtoms s r) : validateScalar s r = .ok (plainRecipe r family width) := by
  have hp := scalar_prefix_from_individual_atoms s r h.loss a
  have ho := plain_option_fields_derive_membership r family width name h
  have ha := plain_adapter_checks_from_individual_atoms s r family width name h a
  have hf : erpContinuationFamily r.initialization = .ok family := by
    rw [h.initialization, erpContinuationFamily, family_text_decodes]
  have hb : r.bnBuffers ≠ .text "online" := by rcases h.buffers with he | he <;> simp [he]
  simp [validateScalar, adapterPrecheck, h.adapter, h.weights, require, hp,
    recipeRemainder, ho, h.nameField, readTextValue, liftCheck, h.nonblank,
    hf, h.affine, h.dropout, h.loss, rejectWhen, compareResult, compareCall,
    atomicCall, a.priorNonzero, h.widthField, positiveInt, ensure,
    h.widthPositive, h.plain, h.noKind, h.noRank, hb, ha, plainRecipe]

theorem explicit_kwargs_from_raw_fields_and_atoms (s : Primitives) (r : RawRecipe)
    (family : Family) (width : ℕ) (name : String) (h : PlainStructure r family width name)
    (a : ScalarAtoms s r) :
    fromKwargsScalar s (Fields.ofList (rawFields r)) = .ok (plainRecipe r family width) := by
  rw [kwargs_reconstruct_before_scalar]
  exact plain_constructor_from_raw_fields_and_atoms s r family width name h a

theorem ERP_fields_derive_plain_structure (f : ERPFields) (h : Conditions f)
    (hf : f.family ≠ .temporal) : PlainStructure f.raw f.family f.width f.name := by
  have hkind : f.kind = none ∧ f.rank = none := by
    rcases h.temporal with ⟨ht, _⟩ | ⟨_, hk, hr⟩
    · exact False.elim (hf ht)
    · exact ⟨hk, hr⟩
  refine ⟨rfl, rfl, h.name, ?_, ?_, rfl, rfl, rfl, rfl, rfl,
    rfl, h.width, hf, ?_, ?_, ?_, ?_⟩
  · rcases h.statistics with hs | hs <;> simp [ERPFields.raw, hs]
  · rcases h.buffers with hs | hs <;> simp [ERPFields.raw, hs]
  · simp [ERPFields.raw, hkind.1]
  · simp [ERPFields.raw, hkind.2]
  · rcases h.maxUpdates with hn | ⟨n, hn, _⟩
    · exact Or.inl hn
    · right; simp [ERPFields.raw, hn, ExactInt]
  · simp [ERPFields.raw, ExactBool]

theorem canonical_kwargs_from_independent_fields_and_atoms (s : Primitives)
    (f : ERPFields) (h : Conditions f) (hf : f.family ≠ .temporal)
    (a : ScalarAtoms s f.raw) : fromKwargsScalar s (recordFields f.raw) = .ok f.recipe := by
  have hkind : f.kind = none ∧ f.rank = none := by
    rcases h.temporal with ⟨ht, _⟩ | ⟨_, hk, hr⟩
    · exact False.elim (hf ht)
    · exact ⟨hk, hr⟩
  have hr := canonical_record_reconstructs_fields f (fun _ => hkind.2)
  have hc := plain_constructor_from_raw_fields_and_atoms s f.raw f.family f.width f.name
    (ERP_fields_derive_plain_structure f h hf) a
  simpa [fromKwargsScalar, hr, liftCheck, plainRecipe, ERPFields.recipe, hkind.1, hkind.2] using hc

theorem temperature_exception_from_single_numpy_atom (s : Primitives) (r : RawRecipe)
    (exception : AtomicException) (ha : r.targetAdapterConfig = .null)
    (hw : r.weights ≠ .text "adapter") (hl : r.candidateLossWeight = .null)
    (hn : numericNonBool r.candidateLossTemperature = true)
    (he : s.numpyIsFinite r.candidateLossTemperature = .error exception) :
    validateScalar s r = .error (.atomic "candidate_loss_temperature" .numpyFinite exception) := by
  have ht := numpy_exception_preserved s "candidate_loss_temperature" false
    r.candidateLossTemperature exception hn he
  simp [validateScalar, adapterPrecheck, ha, hw, require, scalarPrefix,
    hl, loss_none_skips_every_operation, ht, andThen, done]

theorem sequence_initialization_from_hashability_after_atoms (s : Primitives) (r : RawRecipe)
    (xs : Values) (ha : r.targetAdapterConfig = .null) (hw : r.weights ≠ .text "adapter")
    (hl : r.candidateLossWeight = .null) (hi : r.initialization = .sequence xs)
    (a : ScalarAtoms s r) :
    validateScalar s r = .error (.atomic "recipe initialization" .setMembership .typeError) := by
  have hp := scalar_prefix_from_individual_atoms s r hl a
  simp [validateScalar, adapterPrecheck, ha, hw, require, hp, recipeRemainder,
    recipeOptions, hi, sequence_option_raises_before_membership]

theorem object_initialization_from_hashability_after_atoms (s : Primitives) (r : RawRecipe)
    (fs : Fields) (ha : r.targetAdapterConfig = .null) (hw : r.weights ≠ .text "adapter")
    (hl : r.candidateLossWeight = .null) (hi : r.initialization = .object fs)
    (a : ScalarAtoms s r) :
    validateScalar s r = .error (.atomic "recipe initialization" .setMembership .typeError) := by
  have hp := scalar_prefix_from_individual_atoms s r hl a
  simp [validateScalar, adapterPrecheck, ha, hw, require, hp, recipeRemainder,
    recipeOptions, hi, object_option_raises_before_membership]

/-- Fixed observed-operation model, not a universal NumPy implementation. -/
theorem pooled_fixture_scalar_atoms (width : ℕ) :
    ScalarAtoms observedPrimitives (pooledFields width).raw := by
  constructor <;> norm_num [observedPrimitives, observedSmallFinite, decodedCompare,
    numericScalar, numericNonBool, ERPFields.raw, pooledFields]

end
end N2P3.PersonalERPScalarFields
