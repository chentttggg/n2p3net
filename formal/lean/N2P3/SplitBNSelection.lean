import N2P3.SplitBNStatistics

/-!
# Actual SplitBN evaluation selectors and selected buffer families

Source: splitbn.py:398-465,490-534. Fin coordinates follow the actual separate
domain-index dtype/device/shape/range checks. Counters remain signed and the
literal readiness test is equality with zero. A source-domain route is separate
from the predeclared unseen-target aggregate policy. Native tensor operations
and training provenance are not inferred from these decoded coordinates.

Source-first contract: v07_splitbn_lifecycle_contract_01.json.
-/
namespace N2P3.SplitBNSelection
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map
attribute [local simp] Bind.bind Pure.pure Functor.map

inductive Aggregate where
  | pooled | weighted | target
  deriving DecidableEq

inductive EvalSource where
  | sourceDomain | aggregate (source : Aggregate) | unknown
  deriving DecidableEq

inductive Failure where
  | missingSource | conflictingDomainSelectors | missingDomainSelector
  | aggregateHasDomainSelector | invalidSource | policyMismatch
  | uninitializedDomain | uninitializedPooled | missingWeights
  | weightCast | uninitializedWeighted | uninitializedTarget
  deriving DecidableEq

structure Moments (features : ℕ) where
  mean : Fin features → ℝ
  variance : Fin features → ℝ

structure State (domains features : ℕ) where
  domain : Fin domains → Moments features
  domainCount : Fin domains → ℤ
  pooled : Moments features
  pooledCount : ℤ
  target : Moments features
  targetInitialized : Bool

inductive Route (B : Type*) (domains : ℕ) where
  | source (indices : B → Fin domains)
  | aggregate (source : Aggregate)

def resolveRoute {B : Type*} {D : ℕ} (source : Option EvalSource)
    (indices : Option (B → Fin D)) (index : Option (Fin D)) : Except Failure (Route B D) :=
  match source with
  | none => .error .missingSource
  | some .sourceDomain =>
    match indices, index with
    | some _, some _ => .error .conflictingDomainSelectors
    | some rows, none => .ok (.source rows)
    | none, some constant => .ok (.source (fun _ => constant))
    | none, none => .error .missingDomainSelector
  | some other =>
    if indices.isSome || index.isSome then .error .aggregateHasDomainSelector
    else match other with
      | .aggregate policy => .ok (.aggregate policy)
      | _ => .error .invalidSource

def rowsReady {B : Type*} {D F : ℕ} (state : State D F) (indices : B → Fin D) : Prop :=
  ∀ row, state.domainCount (indices row) ≠ 0

def domainsReady {D F : ℕ} (state : State D F) (weights : Fin D → ℝ) : Prop :=
  ∀ domain, 0 < weights domain → state.domainCount domain ≠ 0

def weightedMoments {D F : ℕ} (state : State D F) (weights : Fin D → ℝ) : Moments F :=
  ⟨fun feature => SplitBNStatistics.mean weights (fun domain => (state.domain domain).mean feature),
    fun feature => SplitBNStatistics.variance weights
      (fun domain => (state.domain domain).mean feature)
      (fun domain => (state.domain domain).variance feature)⟩

def weightedSelect {D F : ℕ} (state : State D F) (configured : Option (Fin D → ℝ))
    (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ)) : Except Failure (Moments F) := do
  let declared ← match configured with
    | none => .error .missingWeights
    | some weights => .ok weights
  let effective ← cast declared
  if domainsReady state effective then .ok (weightedMoments state effective)
  else .error .uninitializedWeighted

def aggregateSelect {D F : ℕ} (state : State D F) (declared selected : Aggregate)
    (weights : Option (Fin D → ℝ)) (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ)) :
    Except Failure (Moments F) :=
  if selected ≠ declared then .error .policyMismatch
  else match selected with
    | .pooled => if state.pooledCount = 0 then .error .uninitializedPooled else .ok state.pooled
    | .weighted => weightedSelect state weights cast
    | .target => if state.targetInitialized then .ok state.target else .error .uninitializedTarget

def select {B : Type*} {D F : ℕ} (state : State D F) (declared : Aggregate)
    (weights : Option (Fin D → ℝ)) (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ))
    (route : Route B D) : Except Failure (B → Moments F) :=
  match route with
  | .source indices =>
    if rowsReady state indices then .ok (fun row => state.domain (indices row))
    else .error .uninitializedDomain
  | .aggregate source => (aggregateSelect state declared source weights cast).map (fun values _ => values)

def evaluate {B : Type*} {D F : ℕ} (state : State D F) (declared : Aggregate)
    (weights : Option (Fin D → ℝ)) (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ))
    (source : Option EvalSource) (indices : Option (B → Fin D)) (index : Option (Fin D)) :
    Except Failure (B → Moments F) := do
  let route ← resolveRoute source indices index
  select state declared weights cast route

theorem missing_eval_source_is_refused_before_any_buffer_family {B : Type*} {D : ℕ}
    (indices : Option (B → Fin D)) (index : Option (Fin D)) :
    resolveRoute none indices index = .error .missingSource := rfl

theorem source_domain_refuses_two_simultaneous_selectors {B : Type*} {D : ℕ}
    (indices : B → Fin D) (index : Fin D) :
    resolveRoute (some .sourceDomain) (some indices) (some index) =
      .error .conflictingDomainSelectors := rfl

theorem source_domain_requires_a_selector {B : Type*} {D : ℕ} :
    resolveRoute (B := B) (D := D) (some .sourceDomain) none none =
      .error .missingDomainSelector := rfl

theorem active_row_and_active_domain_counter_guards_are_equivalent
    {B : Type*} {D F : ℕ} (state : State D F) (indices : B → Fin D) :
    rowsReady state indices ↔
      ∀ domain, (∃ row, indices row = domain) → state.domainCount domain ≠ 0 := by
  constructor
  · intro ready domain active
    obtain ⟨row, rfl⟩ := active
    exact ready row
  · intro ready row
    exact ready (indices row) ⟨row, rfl⟩

theorem source_domain_returns_the_actual_per_row_buffer_independently_of_target_policy
    {B : Type*} {D F : ℕ} (state : State D F) (declared : Aggregate)
    (weights : Option (Fin D → ℝ)) (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ))
    (indices : B → Fin D) (ready : ∀ row, state.domainCount (indices row) ≠ 0) :
    evaluate state declared weights cast (some .sourceDomain) (some indices) none =
      .ok (fun row => state.domain (indices row)) := by
  simp [evaluate, resolveRoute, select, rowsReady, ready]

theorem constant_source_domain_selector_uses_the_same_domain_for_each_row
    {B : Type*} {D F : ℕ} (state : State D F) (declared : Aggregate)
    (weights : Option (Fin D → ℝ)) (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ))
    (index : Fin D) (ready : state.domainCount index ≠ 0) :
    evaluate (B := B) state declared weights cast (some .sourceDomain) none (some index) =
      .ok (fun _ => state.domain index) := by
  simp [evaluate, resolveRoute, select, rowsReady, ready]

theorem an_empty_source_eval_batch_checks_no_inactive_domain
    {D F : ℕ} (state : State D F) (declared : Aggregate)
    (weights : Option (Fin D → ℝ)) (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ))
    (indices : Fin 0 → Fin D) :
    select state declared weights cast (.source indices) =
      .ok (fun row => state.domain (indices row)) := by
  have ready : rowsReady state indices := fun row => Fin.elim0 row
  simp [select, ready]

theorem aggregate_source_refuses_domain_indices_before_policy_or_readiness
    {B : Type*} {D : ℕ} (policy : Aggregate) (indices : B → Fin D)
    (index : Option (Fin D)) :
    resolveRoute (some (.aggregate policy)) (some indices) index =
      .error .aggregateHasDomainSelector := by simp [resolveRoute]

theorem unknown_source_with_an_index_rejects_that_index_first
    {B : Type*} {D : ℕ} (index : Fin D) :
    resolveRoute (B := B) (some .unknown) none (some index) =
      .error .aggregateHasDomainSelector := rfl

theorem aggregate_policy_mismatch_does_not_fall_back_to_a_ready_family
    {D F : ℕ} (state : State D F) (declared selected : Aggregate) (different : selected ≠ declared)
    (weights : Option (Fin D → ℝ)) (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ)) :
    aggregateSelect state declared selected weights cast = .error .policyMismatch := by
  simp [aggregateSelect, different]

theorem pooled_guard_is_the_literal_nonzero_counter_test
    {D F : ℕ} (state : State D F) (ready : state.pooledCount ≠ 0)
    (weights : Option (Fin D → ℝ)) (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ)) :
    aggregateSelect state .pooled .pooled weights cast = .ok state.pooled := by
  simp [aggregateSelect, ready]

theorem pooled_zero_counter_is_refused_even_with_initialized_target
    {D F : ℕ} (state : State D F) (unready : state.pooledCount = 0)
    (weights : Option (Fin D → ℝ)) (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ)) :
    aggregateSelect state .pooled .pooled weights cast = .error .uninitializedPooled := by
  simp [aggregateSelect, unready]

theorem target_without_its_initialized_flag_has_no_pooled_fallback
    {D F : ℕ} (state : State D F) (unready : state.targetInitialized = false)
    (weights : Option (Fin D → ℝ)) (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ)) :
    aggregateSelect state .target .target weights cast = .error .uninitializedTarget := by
  simp [aggregateSelect, unready]

theorem initialized_target_selects_exactly_the_stored_target_moments
    {D F : ℕ} (state : State D F) (ready : state.targetInitialized = true)
    (weights : Option (Fin D → ℝ)) (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ)) :
    aggregateSelect state .target .target weights cast = .ok state.target := by
  simp [aggregateSelect, ready]

theorem source_weighted_branch_uses_the_actual_single_cast_and_active_counters
    {D F : ℕ} (state : State D F) (declared effective : Fin D → ℝ)
    (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ))
    (castReply : cast declared = .ok effective)
    (ready : ∀ domain, 0 < effective domain → state.domainCount domain ≠ 0) :
    aggregateSelect state .weighted .weighted (some declared) cast =
      .ok (weightedMoments state effective) := by
  have activeReady : domainsReady state effective := ready
  simp [aggregateSelect, weightedSelect, castReply, activeReady]

theorem source_weighted_missing_weights_are_not_replaced_by_uniform_weights
    {D F : ℕ} (state : State D F)
    (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ)) :
    aggregateSelect state .weighted .weighted none cast = .error .missingWeights := by
  simp [aggregateSelect, weightedSelect]

theorem weighted_selected_variance_retains_the_source_clamp
    {D F : ℕ} (state : State D F) (weights : Fin D → ℝ) (feature : Fin F) :
    0 ≤ (weightedMoments state weights).variance feature :=
  SplitBNStatistics.source_clamp_is_always_nonnegative weights _ _

theorem initialized_target_evaluation_ignores_source_and_pooled_buffer_changes
    {D F : ℕ} (state : State D F) (ready : state.targetInitialized = true)
    (domains : Fin D → Moments F) (counts : Fin D → ℤ) (pooled : Moments F) (pooledCount : ℤ)
    (weights : Option (Fin D → ℝ)) (cast : (Fin D → ℝ) → Except Failure (Fin D → ℝ)) :
    aggregateSelect { state with domain := domains, domainCount := counts, pooled := pooled, pooledCount := pooledCount }
        .target .target weights cast =
      aggregateSelect state .target .target weights cast := by
  simp [aggregateSelect, ready]

end
end N2P3.SplitBNSelection
