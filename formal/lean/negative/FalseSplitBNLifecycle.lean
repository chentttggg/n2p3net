import N2P3.SplitBNTargetState

namespace N2P3.NegativeSplitBNLifecycle
open N2P3.SplitBNSelection N2P3.SplitBNTargetState

def state : State 2 1 where
  domain := fun domain => ⟨fun _ => if domain = 0 then 10 else 20, fun _ => 1⟩
  domainCount := fun domain => if domain = 0 then 1 else 0
  pooled := ⟨fun _ => 40, fun _ => 4⟩
  pooledCount := 1
  target := ⟨fun _ => 90, fun _ => 9⟩
  targetInitialized := false

def castWeights (weights : Fin 2 → ℝ) : Except SplitBNSelection.Failure (Fin 2 → ℝ) :=
  .ok weights

-- A ready pooled buffer does not supply a target-calibration fallback.
example : aggregateSelect state .target .target none castWeights = .ok state.pooled := by
  rw [target_without_its_initialized_flag_has_no_pooled_fallback state rfl none castWeights]
  simp only [reduceCtorEq]

-- A different aggregate family is not a fallback for the configured policy.
example : aggregateSelect state .target .pooled none castWeights = .ok state.pooled := by
  rw [aggregate_policy_mismatch_does_not_fall_back_to_a_ready_family state .target .pooled
    (by decide) none castWeights]
  simp only [reduceCtorEq]

-- Inactive domain1 has counter0, but every selected row belongs to domain0.
example : evaluate (B := Fin 1) state .target none castWeights (some .sourceDomain)
    (some (fun _ => 0)) none = .error .uninitializedDomain := by
  have ready : ∀ _row : Fin 1, state.domainCount 0 ≠ 0 := by
    intro _
    norm_num [state]
  rw [source_domain_returns_the_actual_per_row_buffer_independently_of_target_policy
    state .target none castWeights (fun _ : Fin 1 => 0) ready]
  simp only [reduceCtorEq]

def negativeCountState : State 2 1 := { state with pooledCount := -1 }

-- The actual pooled readiness test is ==0, not <=0.
example : aggregateSelect negativeCountState .pooled .pooled none castWeights =
    .error .uninitializedPooled := by
  rw [pooled_guard_is_the_literal_nonzero_counter_test negativeCountState
    (by norm_num [negativeCountState]) none castWeights]
  simp only [reduceCtorEq]

def initializedState : State 2 1 := { state with targetInitialized := true }
def newMean (_feature : Fin 1) : ℝ := 3
def newVariance (_feature : Fin 1) : ℝ := 2

-- Explicit frozen=True does not authorize replacing an existing target.
example : install .target true (fun _ => .ok (finiteTensor newMean))
    (fun _ => .ok (finiteTensor newVariance)) initializedState =
    .ok { initializedState with target := ⟨newMean, newVariance⟩ } := by
  rw [an_initialized_target_refuses_any_new_tensor_conversions _ _ initializedState rfl]
  simp only [reduceCtorEq]

-- Clear/reset ends that installation epoch; this is not lifetime immutability.
example : install .target true (fun _ => .ok (finiteTensor newMean))
    (fun _ => .ok (finiteTensor newVariance)) (clearTarget initializedState) =
    .error .alreadyInitialized := by
  rw [clear_permits_a_new_install_in_the_same_object_lifetime newMean newVariance
    (by intro _; norm_num [newVariance]) initializedState]
  simp only [reduceCtorEq]

-- Fresh diagnostic mean9 is returned while the old stored mean2 is retained.
example : finalize (S := ℝ) (E := SplitBNTargetState.Failure) .diagnostic 2 9
    (fun _ new => .ok new) = .ok (9, 9) := by
  norm_num [finalize]

-- Mixed initialization is rejected before installation or diagnostics.
example : helperAction [true, false] = .ok .install := by
  rw [mixed_initialization_is_refused_before_a_finalization_action [true, false] (by simp) (by simp)]
  simp only [reduceCtorEq]

end N2P3.NegativeSplitBNLifecycle
