import N2P3.NeuralStrictLoad
import N2P3.SplitBNSelection

/-! NN04 and the post-allocation personal neural control sequence. Actual
audited classes use Module.train's per-child training assignment; no arbitrary
callback purity or restoration of prior mixed child modes is assumed. This
does not claim full forward, input-preprocessing or checkpoint-SplitBN closure.
The architecture record is the value returned by its separately audited field
reader; equality is an executed guard, not a final predictor-equality premise. -/
namespace N2P3.NeuralEvaluationControl

open N2P3.NeuralConstructorConfig N2P3.NeuralRegisteredState N2P3.NeuralStrictLoad
open N2P3.PersonalState N2P3.FrozenState
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map

def neuralKind (kind : String) : Bool := kind="source_initialized_n2p3" || kind="scratch_n2p3"

structure Control where
  trainingModes : List (String × Bool)
  tensors : RuntimeState
  targetEnabled : Option Bool
  autogradEnabled : Bool

def evalControl (control : Control) : Control :=
  {control with trainingModes:=control.trainingModes.map (fun entry => (entry.1,false))}

def constructorControl (modes : List (String × Bool)) (loaded : RuntimeState) (wrapper : Option Wrapper) : Control :=
  ⟨modes,loaded,wrapper.map (fun _ => true),true⟩

def savedFinite (finiteTensor : Tensor → Bool) (source : NamedState) : Bool :=
  source.all (fun entry => finiteTensor entry.2)

inductive RouteFailure where
  | overridePolicy | sourceIndex | ordinarySelectors
  deriving DecidableEq

/- checkpoint._checkpoint_prediction_input resolves an unseen-target policy
before the actual SplitBNSelection evaluator. It never supplies source rows. -/
def targetRoute {D : ℕ} (declared : Option N2P3.SplitBNSelection.Aggregate)
    (requested : Option N2P3.SplitBNSelection.EvalSource) (index : Option (Fin D)) :
    Except RouteFailure (Option N2P3.SplitBNSelection.EvalSource) :=
  match declared with
  | none => .ok requested
  | some policy =>
    if requested.isSome ∧ requested≠some (.aggregate policy) then .error .overridePolicy
    else if index.isSome then .error .sourceIndex
    else .ok (some (.aggregate policy))

def ordinaryRoute {D : ℕ} (sourceRows : Bool)
    (requested : Option N2P3.SplitBNSelection.EvalSource) (index : Option (Fin D)) : Except RouteFailure Unit :=
  if sourceRows || requested.isSome || index.isSome then .error .ordinarySelectors else .ok ()

theorem personal_ordinary_BN_call_has_no_domain_or_aggregate_selector {D : ℕ} :
    ordinaryRoute (D:=D) false none none=.ok () := rfl

theorem ordinary_BN_refuses_a_SplitBN_buffer_selector {D : ℕ}
    (source : N2P3.SplitBNSelection.EvalSource) (index : Option (Fin D)) :
    ordinaryRoute false (some source) index=.error .ordinarySelectors := by simp [ordinaryRoute]

theorem checkpoint_target_call_selects_its_predeclared_aggregate {D : ℕ}
    (policy : N2P3.SplitBNSelection.Aggregate) :
    targetRoute (D:=D) (some policy) none none=.ok (some (.aggregate policy)) := by simp [targetRoute]

theorem checkpoint_target_call_cannot_override_aggregate_policy_with_source_domain {D : ℕ}
    (policy : N2P3.SplitBNSelection.Aggregate) (index : Option (Fin D)) :
    targetRoute (some policy) (some .sourceDomain) index=.error .overridePolicy := by simp [targetRoute]

theorem checkpoint_target_call_refuses_a_source_domain_index {D : ℕ}
    (policy : N2P3.SplitBNSelection.Aggregate) (index : Fin D) :
    targetRoute (some policy) none (some index)=.error .sourceIndex := by simp [targetRoute]

inductive Failure where
  | architecture | state (fault : NeuralStrictLoad.Failure) | inputStatistics | nonfiniteSavedState
  deriving DecidableEq

structure Loaded where
  control : Control
  savedState : NamedState
  plan : Plan
  wrapper : Option Wrapper

def finish (cast : Cast) (finiteTensor : Tensor → Bool) (plan : Plan) (dtype : String)
    (wrapper : Option Wrapper) (modes : List (String × Bool))
    (actualArchitecture expectedArchitecture : Value) (modelSource source : NamedState) : Except Failure Loaded := do
  if actualArchitecture≠expectedArchitecture then .error .architecture else pure ()
  let loaded ← (strictLoad cast (registry dtype plan wrapper) modelSource).mapError Failure.state
  let control := evalControl (constructorControl modes loaded wrapper)
  if inputStatsGuard plan.arguments.channels source=false then .error .inputStatistics else pure ()
  if savedFinite finiteTensor source=false then .error .nonfiniteSavedState else pure ()
  .ok ⟨control,source,plan,wrapper⟩

theorem both_actual_neural_model_kinds_select_the_same_control_branch :
    neuralKind "source_initialized_n2p3"=true ∧ neuralKind "scratch_n2p3"=true := by decide

theorem recursive_eval_sets_each_audited_module_mode_to_false (control : Control) :
    ∀ entry ∈ (evalControl control).trainingModes,entry.2=false := by
  intro entry member
  obtain ⟨original,_,same⟩ := List.mem_map.mp member
  subst entry
  rfl

theorem eval_preserves_all_loaded_values_and_the_target_enable_flag (control : Control) :
    (evalControl control).tensors=control.tensors ∧
      (evalControl control).targetEnabled=control.targetEnabled := by exact ⟨rfl,rfl⟩

theorem eval_does_not_itself_disable_autograd (control : Control) :
    (evalControl control).autogradEnabled=control.autogradEnabled := rfl

theorem repeated_eval_is_idempotent_even_after_mixed_original_child_modes (control : Control) :
    evalControl (evalControl control)=evalControl control := by
  simp [evalControl,List.map_map,Function.comp_def]

theorem adapter_enable_is_restored_from_the_branch_independently_of_saved_tensors
    (modes : List (String × Bool)) (loaded : RuntimeState) (wrapper : Wrapper) :
    (constructorControl modes loaded (some wrapper)).targetEnabled=some true ∧
      (constructorControl modes loaded none).targetEnabled=none := by exact ⟨rfl,rfl⟩

theorem architecture_mismatch_refuses_before_any_state_cast
    (cast : Cast) (finiteTensor : Tensor → Bool) (plan : Plan) (dtype : String)
    (wrapper : Option Wrapper) (modes : List (String × Bool))
    (actual expected : Value) (modelSource source : NamedState) (different : actual≠expected) :
    finish cast finiteTensor plan dtype wrapper modes actual expected modelSource source=.error .architecture := by
  simp [finish,different]

theorem postload_statistics_guard_is_independent_of_the_model_registry
    (channels : ℕ) (source : NamedState) (mean std : Tensor)
    (meanFound : lookupTensor "input_mean" source=some mean)
    (stdFound : lookupTensor "input_std" source=some std) :
    inputStatsGuard channels source=decide (validInputStats channels mean std) := by
  simp [inputStatsGuard,meanFound,stdFound]

theorem global_saved_finite_check_really_checks_every_saved_entry
    (finiteTensor : Tensor → Bool) (source : NamedState) :
    savedFinite finiteTensor source=true ↔ ∀ entry ∈ source,finiteTensor entry.2=true := by
  simp [savedFinite,List.all_eq_true]

/- The per-entry replies below are proved in NeuralStrictLoad from source
lookup, independent shape compatibility and each scalar cast. No complete
strictLoad/constructor/shape/forward result is an input premise. -/
theorem actual_postallocation_control_is_derived_from_registered_entry_operations
    (cast : Cast) (finiteTensor : Tensor → Bool) (plan : Plan) (dtype : String)
    (wrapper : Option Wrapper) (modes : List (String × Bool)) (architecture : Value) (modelSource source : NamedState)
    (values : Slot → RuntimeTensor)
    (each : ∀ slot ∈ registry dtype plan wrapper,loadOne cast slot modelSource=.ok (values slot))
    (keys : noUnexpected (registry dtype plan wrapper) modelSource)
    (stats : inputStatsGuard plan.arguments.channels source=true)
    (finite : ∀ entry ∈ source,finiteTensor entry.2=true) :
    finish cast finiteTensor plan dtype wrapper modes architecture architecture modelSource source=
      .ok ⟨evalControl (constructorControl modes
        ((registry dtype plan wrapper).map (fun slot => (slot.name,values slot))) wrapper),source,plan,wrapper⟩ := by
  have loaded := complete_registered_load_is_derived_from_each_tensor_copy cast modelSource _ values each
  have allFinite := (global_saved_finite_check_really_checks_every_saved_entry finiteTensor source).mpr finite
  simp [finish,strictLoad,keys,loaded,stats,allFinite]

end
end N2P3.NeuralEvaluationControl
