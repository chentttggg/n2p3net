import N2P3.PersonalERPScalar

/-! Eighteen independently fixed wrong alternatives to scalar tags, exceptions,
loop priority, optimization domains and concrete personal-branch composition.
Every example must fail only with one unsolved False goal. -/
namespace N2P3.NegativePersonalERPScalar
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalState N2P3.ERPState
open N2P3.PersonalERPRecipe N2P3.PersonalERPScalar

example : (scalarField observedPrimitives "candidate_loss_temperature" false (.boolean true)).result = .ok () := by
  rw [bool_rejected_before_numpy]
  change (Except.error (.sourceValueError "candidate_loss_temperature") : Result Unit) = .ok ()
  have hn : (Except.error (.sourceValueError "candidate_loss_temperature") : Result Unit) ≠ .ok () := by intro h; cases h
  simp only [hn]

example : (scalarField observedPrimitives "candidate_loss_temperature" false .null).result = .ok () := by
  rw [temperature_none_is_not_skipped]
  change (Except.error (.sourceValueError "candidate_loss_temperature") : Result Unit) = .ok ()
  have hn : (Except.error (.sourceValueError "candidate_loss_temperature") : Result Unit) ≠ .ok () := by intro h; cases h
  simp only [hn]

example : (scalarField observedPrimitives "candidate_loss_weight" true .null).calls =
    [⟨"candidate_loss_weight", .numpyFinite⟩] := by
  rw [loss_none_skips_every_operation]
  change ([] : List Call) = [⟨"candidate_loss_weight", Operation.numpyFinite⟩]
  have hn : ([] : List Call) ≠ [⟨"candidate_loss_weight", .numpyFinite⟩] := by intro h; cases h
  simp only [hn]

example : (scalarField observedPrimitives "candidate_loss_temperature" false huge100).result =
    .error (.sourceValueError "candidate_loss_temperature") := by
  rw [huge_integer_not_treated_as_positive_real]
  change (Except.error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) : Result Unit) =
    .error (.sourceValueError "candidate_loss_temperature")
  have hn : (Except.error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) : Result Unit) ≠
      .error (.sourceValueError "candidate_loss_temperature") := by intro h; cases h
  simp only [hn]

example : observedPrimitives.numpyIsFinite huge100 = .ok true := by
  rw [observed_numpy_huge_typeerror]
  have hn : (Except.error AtomicException.typeError : Except AtomicException Bool) ≠ .ok true := by intro h; cases h
  simp only [hn]

example : observedPrimitives.mathIsFinite huge100 = .error .typeError := by
  rw [observed_math_huge_finite]
  have hn : (Except.ok true : Except AtomicException Bool) ≠ .error .typeError := by intro h; cases h
  simp only [hn]

example : observedPrimitives.mathIsFinite huge400 = .ok true := by
  rw [observed_math_larger_overflow]
  have hn : (Except.error AtomicException.overflowError : Except AtomicException Bool) ≠ .ok true := by intro h; cases h
  simp only [hn]

example : (scalarPrefix observedPrimitives
    { (pooledFields 2).raw with candidateLossTemperature := .integer 0, sourcePriorWeight := huge100 }).result =
    .error (.sourceValueError "candidate_loss_temperature positive") := by
  rw [zero_temperature_checks_prior_first]
  change (Except.error (.atomic "source_prior_weight" .numpyFinite .typeError) : Result Unit) =
    .error (.sourceValueError "candidate_loss_temperature positive")
  have hn : (Except.error (.atomic "source_prior_weight" .numpyFinite .typeError) : Result Unit) ≠
      .error (.sourceValueError "candidate_loss_temperature positive") := by intro h; cases h
  simp only [hn]

example : (scalarPrefix observedPrimitives
    { (pooledFields 2).raw with candidateLossWeight := .integer (-1), candidateLossTemperature := huge100 }).result =
    .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by
  rw [negative_loss_stops_before_huge_temperature]
  have hn : (Except.error (.sourceValueError "candidate_loss_weight") : Result Unit) ≠
      .error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) := by intro h; cases h
  simp only [hn]

example : optimization observedPrimitives
    { (pooledFields 2).raw with epochs := .boolean true, batchSize := .boolean true, lr := .boolean true } =
    .error (.sourceValueError "adapter epochs") := by
  rw [positive_bool_optimization_is_not_strict_integer_validation]
  have hn : (Except.ok () : Result Unit) ≠ .error (.sourceValueError "adapter epochs") := by intro h; cases h
  simp only [hn]

example : optimization observedPrimitives
    { (pooledFields 2).raw with epochs := huge100, batchSize := huge100, lr := huge100 } =
    .error (.atomic "adapter epochs" .numpyFinite .typeError) := by
  rw [huge_positive_optimization_uses_no_finite_operation]
  have hn : (Except.ok () : Result Unit) ≠ .error (.atomic "adapter epochs" .numpyFinite .typeError) := by intro h; cases h
  simp only [hn]

example : readBranchScalar observedPrimitives fixturePrimitives fixtureReference fixturePhysical
    "target_only_structured_erp" (.boolean false)
    (Fields.ofList (rawFields (pooledFields 2).raw))
    (plainRecord .pooled 2 4 2 5 (.text "none")) fixtureState =
    .error (.unrefined "whole recipe success") := by
  rw [legal_scalar_branch_nonempty]
  have hn : (Except.ok ⟨fixtureRecipe, [0, 2], fixtureModel⟩ : Result Bound) ≠
      .error (.unrefined "whole recipe success") := by intro h; cases h
  simp only [hn]

example : readBranchScalar observedPrimitives fixturePrimitives fixtureReference fixturePhysical
    "target_only_structured_erp" (.boolean false)
    (Fields.ofList (rawFields hugeTemperatureRaw))
    (plainRecord .pooled 2 4 2 5 (.text "none")) fixtureState =
    .error (.sourceValueError "candidate_loss_temperature") := by
  rw [huge_temperature_at_actual_branch_position]
  have hn : (Except.error (.atomic "candidate_loss_temperature" .numpyFinite .typeError) : Result Bound) ≠
      .error (.sourceValueError "candidate_loss_temperature") := by intro h; cases h
  simp only [hn]

example : validateScalar observedPrimitives
    { (pooledFields 2).raw with inputStatistics := .sequence .nil } =
    .error (.sourceValueError "recipe input_statistics") := by
  rw [list_recipe_option_preserves_typeerror]
  have hn : (Except.error (.atomic "recipe input_statistics" .setMembership .typeError) : Result Recipe) ≠
      .error (.sourceValueError "recipe input_statistics") := by intro h; cases h
  simp only [hn]

example : validateScalar observedPrimitives
    { (pooledFields 2).raw with inputStatistics := .object .nil } =
    .error (.sourceValueError "recipe input_statistics") := by
  rw [dict_recipe_option_preserves_typeerror]
  have hn : (Except.error (.atomic "recipe input_statistics" .setMembership .typeError) : Result Recipe) ≠
      .error (.sourceValueError "recipe input_statistics") := by intro h; cases h
  simp only [hn]

example : validateScalar observedPrimitives
    { (pooledFields 2).raw with inputStatistics := .text "unsupported_probe_value" } =
    .error (.atomic "recipe input_statistics" .setMembership .typeError) := by
  rw [hashable_wrong_string_option_is_valueerror]
  have hn : (Except.error (.sourceValueError "recipe input_statistics") : Result Recipe) ≠
      .error (.atomic "recipe input_statistics" .setMembership .typeError) := by intro h; cases h
  simp only [hn]

example : recipeRemainder observedPrimitives
    { (temporalFields "fixed_cosine" 1).raw with temporalBasisKind := .sequence .nil } =
    .error (.sourceValueError "temporal declaration") := by
  rw [temporal_kind_container_is_typeerror]
  have hn : (Except.error (.atomic "temporal declaration" .setMembership .typeError) : Result Recipe) ≠
      .error (.sourceValueError "temporal declaration") := by intro h; cases h
  simp only [hn]

example : adapterChecks observedPrimitives
    { (pooledFields 2).raw with bnBuffers := .sequence .nil } =
    .error (.sourceValueError "adapter batchnorm_buffers") := by
  rw [adapter_option_container_is_typeerror]
  have hn : (Except.error (.atomic "adapter batchnorm_buffers" .setMembership .typeError) : Result Unit) ≠
      .error (.sourceValueError "adapter batchnorm_buffers") := by intro h; cases h
  simp only [hn]

end N2P3.NegativePersonalERPScalar
