import N2P3.NetworkReadoutGeometry
import N2P3.PersonalTargetRecipeFields

/-! NN01's normalized finite-scalar constructor slice. Fields are the actual
15 architecture kwargs after the individually audited Python list/int/float
operations, not an assumed N2P3Net/TargetRecipe result. This source-domain
record retains every kwarg and derives local guards/geometry/clock inputs.
Unrestricted JSON scalar conversion remains the explicit operation boundary;
full value forward is the distinct V07 dependency. -/
namespace N2P3.NeuralConstructorConfig

open N2P3.NetworkReadoutGeometry
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map

structure Residual where
  bottleneck : ℕ
  kernel : ℕ
  maxResidual : ℝ

structure Architecture where
  temporalFilters : ℕ
  temporalKernel : ℕ
  temporalDilation : ℕ
  spatialDepth : ℕ
  stPool : ℕ
  mstKernels : List ℕ
  mstDilations : Option (List ℕ)
  mstPerScale : ℕ
  mstPool : ℕ
  dropout : ℝ
  spatialMaxNorm : ℝ
  interactionRank : ℕ
  hiddenFeatures : ℕ
  observedChannelNorm : Bool
  featureResidual : Option Residual

def resolvedDilations (a : Architecture) : List ℕ :=
  a.mstDilations.getD (a.mstKernels.map (fun _ => 1))

def normalizedArchitecture (a : Architecture) : Architecture :=
  {a with mstDilations:=some (resolvedDilations a)}

def localGuards (a : Architecture) : Prop :=
  0<a.temporalFilters ∧ 0<a.spatialDepth ∧
  3≤a.temporalKernel ∧ a.temporalKernel%2=1 ∧ 0<a.temporalDilation ∧
  a.mstKernels≠[] ∧ (∀ kernel ∈ a.mstKernels,3≤kernel ∧ kernel%2=1) ∧
  (resolvedDilations a).length=a.mstKernels.length ∧
  (∀ dilation ∈ resolvedDilations a,0<dilation) ∧
  0<a.mstPerScale ∧ 0<a.stPool ∧ 0<a.mstPool ∧
  0≤a.dropout ∧ a.dropout<1 ∧ 0<a.spatialMaxNorm ∧
  0<a.interactionRank ∧ 0<a.hiddenFeatures ∧
  (∀ residual,a.featureResidual=some residual →
    0<residual.bottleneck ∧ 0<residual.kernel ∧ residual.kernel%2=1 ∧ 0<residual.maxResidual)

structure Physical where
  channels : ℕ
  times : ℕ
  sfreq : ℝ
  tminMs : ℝ

/- Literal N2P3Net keyword arguments. SplitBN arguments are not supplied by
this personal constructor; latency inputs are its real default arguments. -/
structure ModelArguments where
  architecture : Architecture
  channels : ℕ
  times : ℕ
  sfreq : ℝ
  tminSeconds : ℝ
  pooling : Mode
  splitBN : Option ℕ
  evidence : ℝ × ℝ
  reference : ℝ × ℝ
  offsets : List ℝ
  temperature : ℝ

def modelArguments (a : Architecture) (p : Physical) (mode : Mode) : ModelArguments :=
  ⟨normalizedArchitecture a,p.channels,p.times,p.sfreq,p.tminMs/1000,mode,none,
    (250,600),(-200,0),[-100,-50,0,50,100],1/2⟩

def spatialChannels (a : Architecture) : ℕ := a.temporalFilters*a.spatialDepth
def featureChannels (a : Architecture) : ℕ := a.mstKernels.length*a.mstPerScale

structure FeatureClock where
  sampleRate : ℝ
  startSeconds : ℝ
  times : ℕ

def featureClock (a : Architecture) (p : Physical) : FeatureClock :=
  ⟨p.sfreq/(a.stPool : ℝ),p.tminMs/1000+((a.stPool-1 : ℕ) : ℝ)/(2*p.sfreq),p.times/a.stPool⟩

inductive Failure where
  | configuration | physical | geometry (fault : NetworkReadoutGeometry.Failure)
  deriving DecidableEq

structure Plan where
  arguments : ModelArguments
  stTimes : ℕ
  classifierWidth : ℕ
  clock : FeatureClock

def constructPlan (a : Architecture) (p : Physical) (mode : Mode) : Except Failure Plan := do
  if ¬localGuards a then .error .configuration else pure ()
  if p.channels=0 ∨ p.times=0 then .error .physical else pure ()
  let dimensions ← (geometryPlan mode (some p.times) (featureChannels a) a.stPool a.mstPool).mapError Failure.geometry
  .ok ⟨modelArguments a p mode,dimensions.1.getD 0,dimensions.2,featureClock a p⟩

theorem absent_dilation_field_creates_one_per_actual_kernel (a : Architecture) (absent : a.mstDilations=none) :
    resolvedDilations a=a.mstKernels.map (fun _ => 1) ∧ (resolvedDilations a).length=a.mstKernels.length := by
  simp [resolvedDilations,absent]

theorem explicit_dilation_field_keeps_its_order (a : Architecture) (values : List ℕ)
    (present : a.mstDilations=some values) : resolvedDilations a=values := by
  simp [resolvedDilations,present]

theorem physical_input_fields_feed_the_literal_constructor_arguments (a : Architecture) (p : Physical) (mode : Mode) :
    (modelArguments a p mode).architecture=normalizedArchitecture a ∧
    (modelArguments a p mode).channels=p.channels ∧ (modelArguments a p mode).times=p.times ∧
    (modelArguments a p mode).sfreq=p.sfreq ∧ (modelArguments a p mode).tminSeconds=p.tminMs/1000 ∧
    (modelArguments a p mode).pooling=mode ∧ (modelArguments a p mode).splitBN=none := by
  exact ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl⟩

theorem all_architecture_kwargs_follow_the_source_dilation_normalization (a : Architecture) (p : Physical) (mode : Mode) :
    (modelArguments a p mode).architecture={a with mstDilations:=some (resolvedDilations a)} := rfl

theorem missing_persistent_latency_buffers_do_not_remove_their_constructor_dependencies
    (a : Architecture) (p : Physical) :
    (modelArguments a p .latencyContrast).evidence=(250,600) ∧
    (modelArguments a p .latencyContrast).reference=(-200,0) ∧
    (modelArguments a p .latencyContrast).offsets=[-100,-50,0,50,100] ∧
    (modelArguments a p .latencyContrast).temperature=1/2 ∧
    (featureClock a p).sampleRate=p.sfreq/(a.stPool : ℝ) ∧
    (featureClock a p).startSeconds=p.tminMs/1000+((a.stPool-1 : ℕ) : ℝ)/(2*p.sfreq) := by
  exact ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

theorem actual_plan_follows_from_independent_fields_and_local_window_guards
    (a : Architecture) (p : Physical) (mode : Mode) (guards : localGuards a)
    (channels : 0<p.channels) (times : 0<p.times)
    (completeST : a.stPool≤p.times) (completeMS : mode=.msFlatten → a.stPool*a.mstPool≤p.times) :
    constructPlan a p mode=.ok ⟨modelArguments a p mode,p.times/a.stPool,
      headWidth mode (featureChannels a) (p.times/a.stPool) a.mstPool,featureClock a p⟩ := by
  have nonempty : 0<a.mstKernels.length := List.length_pos_iff.mpr guards.2.2.2.2.2.1
  have perScale : 0<a.mstPerScale := guards.2.2.2.2.2.2.2.2.2.1
  have stPool : 0<a.stPool := guards.2.2.2.2.2.2.2.2.2.2.1
  have msPool : 0<a.mstPool := guards.2.2.2.2.2.2.2.2.2.2.2.1
  have dims := (all_known_time_readout_plans_and_positive_head_widths_follow_from_local_windows mode
    p.times (featureChannels a) a.stPool a.mstPool (Nat.mul_pos nonempty perScale)
    stPool msPool completeST completeMS).1
  simp [constructPlan,guards,Nat.ne_of_gt channels,Nat.ne_of_gt times,dims]

/- The full recipe constructor below is the accepted per-field graph. The
plain branch never invokes it. It is not a whole-network constructor oracle. -/
def adapterConfig (primitives : N2P3.PersonalRecipePython.Primitives)
    (kwargs : N2P3.PersonalRecipePython.Fields) :
    N2P3.PersonalRecipePython.Result (Option N2P3.PersonalTargetRecipe.FeatureConfig) :=
  if kwargs.get "weights"=some (.text "adapter") then
    (N2P3.PersonalTargetRecipe.fromKwargs primitives kwargs).map (·.adapter)
  else .ok none

theorem plain_weight_branch_does_not_invoke_the_recipe_constructor
    (primitives : N2P3.PersonalRecipePython.Primitives)
    (kwargs : N2P3.PersonalRecipePython.Fields) (plain : kwargs.get "weights"≠some (.text "adapter")) :
    adapterConfig primitives kwargs=.ok none := by simp [adapterConfig,plain]

theorem adapter_branch_uses_the_existing_full_field_constructor
    (primitives : N2P3.PersonalRecipePython.Primitives) (fields : N2P3.PersonalTargetRecipeFields.Arguments)
    (config : Option N2P3.PersonalTargetRecipe.FeatureConfig) (value : N2P3.PersonalRecipePython.Value)
    (guards : N2P3.PersonalTargetRecipeFields.Conditions fields)
    (atoms : N2P3.PersonalTargetRecipeFields.AtomicSuccess primitives fields)
    (adapter : N2P3.PersonalTargetRecipeFields.AdapterFields primitives fields.raw config)
    (json : fields.raw.adapter=.raw value) (weights : fields.weights="adapter") :
    adapterConfig primitives (N2P3.PersonalTargetRecipe.allRecordFields fields.raw)=.ok config := by
  have selected : (N2P3.PersonalTargetRecipe.allRecordFields fields.raw).get "weights"=some (.text "adapter") := by
    simp [N2P3.PersonalTargetRecipe.allRecordFields,N2P3.PersonalRecipePython.Fields.ofList,
      N2P3.PersonalRecipePython.Fields.get,N2P3.PersonalTargetRecipeFields.Arguments.raw,weights]
  rw [adapterConfig,if_pos selected,
    N2P3.PersonalTargetRecipeFields.complete_json_kwargs_constructor_from_fields primitives fields config value
      guards atoms adapter json]
  rfl

theorem independently_constructed_target_adapter_fields_are_unindexed
    (primitives : N2P3.PersonalRecipePython.Primitives) (raw : N2P3.PersonalTargetRecipe.Raw)
    (config : N2P3.PersonalTargetRecipe.FeatureConfig)
    (fields : N2P3.PersonalTargetRecipeFields.AdapterFields primitives raw (some config)) :
    config.sourceCount=0 ∧ config.privateCount=0 := by
  cases fields <;> exact ⟨by assumption,by assumption⟩

end
end N2P3.NeuralConstructorConfig
