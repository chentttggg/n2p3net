import N2P3.SplitBNSelection
import Mathlib.Tactic.NormNum

/-!
# Explicit frozen target installation and helper initialization phases

Source: splitbn.py:259-322,585-607,714-729. Conversion replies represent the
individual mean and variance tensor operations, including shape and finite
observations. Exact successful copies update target fields only. Native copy
exceptions and multi-module atomicity are not certified. Helper finalization
starts after its forward/capture pass; arbitrary callbacks are not assumed pure.

Source-first contract: v07_splitbn_lifecycle_contract_01.json.
-/
namespace N2P3.SplitBNTargetState
open N2P3.SplitBNSelection
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map

inductive Failure where
  | wrongPolicy | notExplicitlyFrozen | alreadyInitialized
  | meanCast | varianceCast | shape | nonfinite | negativeVariance
  | noSites | partiallyInitialized
  deriving DecidableEq

structure TensorReply (features : ℕ) where
  values : Fin features → ℝ
  shapeCorrect : Bool
  finite : Bool

def install {D F : ℕ} (policy : Aggregate) (explicitlyFrozen : Bool)
    (meanCast varianceCast : Unit → Except Failure (TensorReply F))
    (state : State D F) : Except Failure (State D F) :=
  if policy ≠ .target then .error .wrongPolicy
  else if explicitlyFrozen ≠ true then .error .notExplicitlyFrozen
  else if state.targetInitialized then .error .alreadyInitialized
  else do
    let mean ← meanCast ()
    let variance ← varianceCast ()
    if mean.shapeCorrect && variance.shapeCorrect then
      if mean.finite && variance.finite then
        if ∀ feature, 0 ≤ variance.values feature then
          .ok { state with target := ⟨mean.values, variance.values⟩, targetInitialized := true }
        else .error .negativeVariance
      else .error .nonfinite
    else .error .shape

def clearTarget {D F : ℕ} (state : State D F) : State D F :=
  { state with target := ⟨fun _ => 0, fun _ => 1⟩, targetInitialized := false }

def finiteTensor {F : ℕ} (values : Fin F → ℝ) : TensorReply F := ⟨values, true, true⟩

theorem wrong_policy_is_refused_before_freeze_initialization_or_casts
    {D F : ℕ} (policy : Aggregate) (different : policy ≠ .target) (frozen : Bool)
    (meanCast varianceCast : Unit → Except Failure (TensorReply F)) (state : State D F) :
    install policy frozen meanCast varianceCast state = .error .wrongPolicy := by
  simp [install, different]

theorem explicit_freeze_guard_precedes_already_initialized_and_cast_guards
    {D F : ℕ} (meanCast varianceCast : Unit → Except Failure (TensorReply F)) (state : State D F) :
    install .target false meanCast varianceCast state = .error .notExplicitlyFrozen := by
  simp [install]

theorem an_initialized_target_refuses_any_new_tensor_conversions
    {D F : ℕ} (meanCast varianceCast : Unit → Except Failure (TensorReply F))
    (state : State D F) (initialized : state.targetInitialized = true) :
    install .target true meanCast varianceCast state = .error .alreadyInitialized := by
  simp [install, initialized]

theorem mean_conversion_failure_precedes_variance_conversion_and_shape_checks
    {D F : ℕ} (varianceCast : Unit → Except Failure (TensorReply F))
    (state : State D F) (uninitialized : state.targetInitialized = false) :
    install .target true (fun _ => .error .meanCast) varianceCast state = .error .meanCast := by
  simp [install, uninitialized]

theorem both_tensor_conversions_precede_the_first_shape_guard
    {D F : ℕ} (mean : TensorReply F) (state : State D F)
    (uninitialized : state.targetInitialized = false) :
    install .target true (fun _ => .ok mean) (fun _ => .error .varianceCast) state =
      .error .varianceCast := by
  simp [install, uninitialized]

theorem bad_shape_is_refused_before_finiteness_or_variance_values
    {D F : ℕ} (mean variance : TensorReply F) (shape : mean.shapeCorrect = false)
    (state : State D F) (uninitialized : state.targetInitialized = false) :
    install .target true (fun _ => .ok mean) (fun _ => .ok variance) state = .error .shape := by
  simp [install, uninitialized, shape]

theorem nonfinite_mean_is_refused_after_valid_shapes
    {D F : ℕ} (mean variance : TensorReply F)
    (meanShape : mean.shapeCorrect = true) (varianceShape : variance.shapeCorrect = true)
    (nonfinite : mean.finite = false) (state : State D F)
    (uninitialized : state.targetInitialized = false) :
    install .target true (fun _ => .ok mean) (fun _ => .ok variance) state = .error .nonfinite := by
  simp [install, uninitialized, meanShape, varianceShape, nonfinite]

theorem one_negative_variance_coordinate_refuses_installation
    {D F : ℕ} (mean variance : Fin F → ℝ) (feature : Fin F) (negative : variance feature < 0)
    (state : State D F) (uninitialized : state.targetInitialized = false) :
    install .target true (fun _ => .ok (finiteTensor mean))
      (fun _ => .ok (finiteTensor variance)) state = .error .negativeVariance := by
  have notNonnegative : ¬ ∀ index, 0 ≤ variance index := by
    intro all
    exact (not_le_of_gt negative) (all feature)
  simp [install, finiteTensor, uninitialized, notNonnegative]

theorem exact_first_install_sets_only_the_actual_cast_target_values_and_flag
    {D F : ℕ} (mean variance : Fin F → ℝ) (nonnegative : ∀ feature, 0 ≤ variance feature)
    (state : State D F) (uninitialized : state.targetInitialized = false) :
    install .target true (fun _ => .ok (finiteTensor mean))
      (fun _ => .ok (finiteTensor variance)) state =
      .ok { state with target := ⟨mean, variance⟩, targetInitialized := true } := by
  simp [install, finiteTensor, uninitialized, nonnegative]

theorem first_install_then_explicit_target_selection_uses_the_installed_moments
    {D F : ℕ} (mean variance : Fin F → ℝ) (nonnegative : ∀ feature, 0 ≤ variance feature)
    (state : State D F) (uninitialized : state.targetInitialized = false)
    (weights : Option (Fin D → ℝ))
    (cast : (Fin D → ℝ) → Except SplitBNSelection.Failure (Fin D → ℝ)) :
    install .target true (fun _ => .ok (finiteTensor mean))
        (fun _ => .ok (finiteTensor variance)) state =
        .ok { state with target := ⟨mean, variance⟩, targetInitialized := true } ∧
      aggregateSelect { state with target := ⟨mean, variance⟩, targetInitialized := true }
        .target .target weights cast = .ok ⟨mean, variance⟩ := by
  exact ⟨exact_first_install_sets_only_the_actual_cast_target_values_and_flag mean variance nonnegative
    state uninitialized, by simp [aggregateSelect]⟩

theorem a_successful_install_blocks_every_subsequent_install_until_a_clear
    {D F : ℕ} (mean variance : Fin F → ℝ) (state : State D F)
    (nextMean nextVariance : Unit → Except Failure (TensorReply F)) :
    install .target true nextMean nextVariance
      { state with target := ⟨mean, variance⟩, targetInitialized := true } =
      .error .alreadyInitialized := by simp [install]

theorem clearing_resets_only_target_moments_and_the_initialization_flag
    {D F : ℕ} (state : State D F) :
    (clearTarget state).targetInitialized = false ∧
      (clearTarget state).target.mean = (fun _ => 0) ∧
      (clearTarget state).target.variance = (fun _ => 1) ∧
      (clearTarget state).domain = state.domain ∧
      (clearTarget state).domainCount = state.domainCount ∧
      (clearTarget state).pooled = state.pooled ∧
      (clearTarget state).pooledCount = state.pooledCount := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem clear_permits_a_new_install_in_the_same_object_lifetime
    {D F : ℕ} (mean variance : Fin F → ℝ) (nonnegative : ∀ feature, 0 ≤ variance feature)
    (state : State D F) :
    install .target true (fun _ => .ok (finiteTensor mean))
      (fun _ => .ok (finiteTensor variance)) (clearTarget state) =
      .ok { clearTarget state with target := ⟨mean, variance⟩, targetInitialized := true } := by
  exact exact_first_install_sets_only_the_actual_cast_target_values_and_flag mean variance nonnegative
    (clearTarget state) rfl

inductive HelperAction where
  | install | diagnostic
  deriving DecidableEq

/-- Source booleans any/all are represented by membership in the actual finite
initialization snapshot, after the separate module-policy/index guards. -/
def helperAction (flags : List Bool) : Except Failure HelperAction :=
  if flags = [] then .error .noSites
  else if true ∈ flags then
    if false ∈ flags then .error .partiallyInitialized else .ok .diagnostic
  else .ok .install

def helperPolicies (sites : List (Aggregate × Bool)) : Except Failure HelperAction :=
  if sites = [] then .error .noSites
  else if ∃ site ∈ sites, site.1 ≠ .target then .error .wrongPolicy
  else helperAction (sites.map Prod.snd)

theorem helper_rejects_non_target_site_policies_before_initialization_patterns
    (sites : List (Aggregate × Bool)) (nonempty : sites ≠ [])
    (wrong : ∃ site ∈ sites, site.1 ≠ .target) :
    helperPolicies sites = .error .wrongPolicy := by simp [helperPolicies, nonempty, wrong]

theorem mixed_initialization_is_refused_before_a_finalization_action
    (flags : List Bool) (initialized : true ∈ flags) (uninitialized : false ∈ flags) :
    helperAction flags = .error .partiallyInitialized := by
  have nonempty : flags ≠ [] := by intro empty; simp [empty] at initialized
  simp [helperAction, nonempty, initialized, uninitialized]

theorem all_uninitialized_sites_request_the_initial_install
    (flags : List Bool) (nonempty : flags ≠ []) (allUninitialized : ∀ flag ∈ flags, flag = false) :
    helperAction flags = .ok .install := by
  have noInitialized : true ∉ flags := by
    intro present
    have impossible := allUninitialized true present
    contradiction
  simp [helperAction, nonempty, noInitialized]

theorem all_initialized_sites_request_diagnostics_only
    (flags : List Bool) (nonempty : flags ≠ []) (allInitialized : ∀ flag ∈ flags, flag = true) :
    helperAction flags = .ok .diagnostic := by
  have noUninitialized : false ∉ flags := by
    intro present
    have impossible := allInitialized false present
    contradiction
  have initialized : true ∈ flags := by
    cases flags with
    | nil => contradiction
    | cons head tail =>
      have headTrue := allInitialized head (by simp)
      simp [headTrue]
  simp [helperAction, nonempty, initialized, noUninitialized]

/-- The post-forward finalization branch, parameterized by the real install
loop. A diagnostic branch never invokes it, even if it would fail or mutate.
The given stored state is the post-forward state; no arbitrary forward purity
or global before/after network read-only premise is hidden here. -/
def finalize {S E A : Type*} (action : HelperAction) (stored : S) (computed : A)
    (installer : S → A → Except E S) : Except E (S × A) :=
  match action with
  | .diagnostic => .ok (stored, computed)
  | .install =>
    match installer stored computed with
    | .error error => .error error
    | .ok next => .ok (next, computed)

theorem diagnostic_finalization_retains_stored_state_and_reports_new_diagnostics
    {S E A : Type*} (stored : S) (computed : A) (installer : S → A → Except E S) :
    finalize .diagnostic stored computed installer = .ok (stored, computed) := rfl

theorem nonempty_already_initialized_helper_has_no_final_install_call
    {S E A : Type*} (flags : List Bool) (nonempty : flags ≠ [])
    (allInitialized : ∀ flag ∈ flags, flag = true)
    (stored : S) (computed : A) (installer : S → A → Except E S) :
    (helperAction flags).map (fun action => finalize action stored computed installer) =
      .ok (.ok (stored, computed)) := by
  rw [all_initialized_sites_request_diagnostics_only flags nonempty allInitialized]
  rfl

theorem diagnostic_values_are_not_asserted_to_equal_frozen_installed_values :
    finalize (S := ℝ) (E := Failure) .diagnostic 2 9 (fun _ new => .ok new) =
      .ok (2, 9) ∧ (2 : ℝ) ≠ 9 := by norm_num [finalize]

end
end N2P3.SplitBNTargetState
