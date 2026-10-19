import N2P3.NeuralConstructorConfig

/-! Actual personal ordinary-BN registration graph, from individual Conv,
Linear, BatchNorm and residual-adapter registrations. Names, shapes, dtype
roles and legacy-counter status are retained for every persistent entry.
Native allocations and default floating dtype are explicit interfaces. -/
namespace N2P3.NeuralRegisteredState

open N2P3.NeuralConstructorConfig N2P3.NetworkReadoutGeometry
noncomputable section

inductive Role where
  | parameter | buffer
  deriving DecidableEq

structure Slot where
  name : String
  role : Role
  dtype : String
  shape : List ℕ
  legacyCounter : Bool := false
  deriving DecidableEq

def parameter (dtype name : String) (shape : List ℕ) : Slot := ⟨name,.parameter,dtype,shape,false⟩
def buffer (dtype name : String) (shape : List ℕ) : Slot := ⟨name,.buffer,dtype,shape,false⟩

def prefixSlot (tag : String) (slot : Slot) : Slot := {slot with name:=tag++slot.name}

def ordinaryBN (dtype tag : String) (features : ℕ) : List Slot :=
  [parameter dtype (tag++"weight") [features],parameter dtype (tag++"bias") [features],
   buffer dtype (tag++"running_mean") [features],buffer dtype (tag++"running_var") [features],
   ⟨tag++"num_batches_tracked",.buffer,"torch.int64",[],true⟩]

def residual (dtype tag : String) (channels bottleneck kernel : ℕ) : List Slot :=
  [parameter dtype (tag++"depthwise.weight") [channels,1,kernel],
   parameter dtype (tag++"down.weight") [bottleneck,channels,1],parameter dtype (tag++"down.bias") [bottleneck],
   parameter dtype (tag++"up.weight") [channels,bottleneck,1],parameter dtype (tag++"up.bias") [channels],
   parameter "torch.float32" (tag++"gate") []]

def branch (dtype : String) (index spatial perScale kernel : ℕ) : List Slot :=
  let tag := "mst_branches."++toString index++"."
  [parameter dtype (tag++"depthwise.weight") [spatial,1,kernel],
   parameter dtype (tag++"pointwise.weight") [perScale,spatial,1]]++ordinaryBN dtype (tag++"norm.") perScale

def head (dtype : String) (mode : Mode) (width rank hidden : ℕ) : List Slot :=
  match mode with
  | .quadraticUnfold =>
    [parameter dtype "classifier.linear.weight" [2,width],parameter dtype "classifier.linear.bias" [2],
     parameter dtype "classifier.left.weight" [rank,width],parameter dtype "classifier.right.weight" [rank,width],
     parameter dtype "classifier.quadratic_output.weight" [2,rank]]
  | .mlpUnfold =>
    [parameter dtype "classifier.linear.weight" [2,width],parameter dtype "classifier.linear.bias" [2],
     parameter dtype "classifier.hidden.weight" [hidden,width],parameter dtype "classifier.hidden.bias" [hidden],
     parameter dtype "classifier.nonlinear_output.weight" [2,hidden]]
  | _ => [parameter dtype "classifier.weight" [2,width],parameter dtype "classifier.bias" [2]]

def plain (dtype : String) (plan : Plan) : List Slot :=
  let a := plan.arguments.architecture
  let spatial := spatialChannels a
  [parameter dtype "st_temporal.weight" [a.temporalFilters,1,1,a.temporalKernel]]++
  ordinaryBN dtype "st_temporal_norm." a.temporalFilters++
  [parameter dtype "st_spatial.weight" [spatial,1,plan.arguments.channels,1]]++
  ordinaryBN dtype "st_spatial_norm." spatial++
  ((a.mstKernels.zipIdx).flatMap (fun pair => branch dtype pair.2 spatial a.mstPerScale pair.1))++
  (if plan.arguments.pooling=.latencyContrast then
    (List.range a.mstKernels.length).map (fun index =>
      parameter dtype ("pool."++toString index++".latency_query") [a.mstPerScale]) else [])++
  head dtype plan.arguments.pooling plan.classifierWidth a.interactionRank a.hiddenFeatures++
  (match a.featureResidual with
   | none => []
   | some config => residual dtype "feature_residual." (featureChannels a) config.bottleneck config.kernel)

structure Wrapper where
  configuration : N2P3.PersonalTargetRecipe.FeatureConfig
  sourceUnindexed : configuration.sourceCount=0
  privateUnindexed : configuration.privateCount=0

def Wrapper.invariantBottleneck (wrapper : Wrapper) : ℕ := wrapper.configuration.invariant.toNat
def Wrapper.privateChannels (wrapper : Wrapper) : ℕ := wrapper.configuration.privateChannels.toNat
def Wrapper.targetBottleneck (wrapper : Wrapper) : ℕ := wrapper.configuration.target.toNat
def Wrapper.targetKernel (wrapper : Wrapper) : ℕ := wrapper.configuration.kernel.toNat

def unindexedWrapper (config : N2P3.PersonalTargetRecipe.FeatureConfig)
    (source : config.sourceCount=0) (privateCount : config.privateCount=0) : Wrapper :=
  ⟨config,source,privateCount⟩

theorem wrapper_retains_all_value_and_control_fields_of_the_constructed_recipe
    (config : N2P3.PersonalTargetRecipe.FeatureConfig)
    (source : config.sourceCount=0) (privateCount : config.privateCount=0) :
    (unindexedWrapper config source privateCount).configuration=config := rfl

/- Full TargetRecipe construction requires unindexed source/private counts0.
Thus its wrapper creates no private_domain_head or source_domain_residuals. -/
def wrapped (dtype : String) (plan : Plan) (config : Wrapper) : List Slot :=
  (plain dtype plan).map (prefixSlot "core.")++
  residual dtype "invariant_adapter." (featureChannels plan.arguments.architecture) config.invariantBottleneck 1++
  [parameter dtype "private_projection.weight" [config.privateChannels,featureChannels plan.arguments.architecture,1],
   parameter dtype "private_projection.bias" [config.privateChannels]]++
  residual dtype "target_adapter." (featureChannels plan.arguments.architecture) config.targetBottleneck config.targetKernel

def registry (dtype : String) (plan : Plan) (wrapper : Option Wrapper) : List Slot :=
  match wrapper with | none => plain dtype plan | some config => wrapped dtype plan config

theorem ordinary_BN_table_contains_affine_moments_and_counter (dtype tag : String) (features : ℕ) :
    (ordinaryBN dtype tag features).length=5 ∧
    (ordinaryBN dtype tag features).map (·.name)=
      [tag++"weight",tag++"bias",tag++"running_mean",tag++"running_var",tag++"num_batches_tracked"] := by exact ⟨rfl,rfl⟩

theorem ordinary_BN_counter_is_the_only_legacy_insertable_entry (dtype tag : String) (features : ℕ) :
    (ordinaryBN dtype tag features).filter (·.legacyCounter)=
      [⟨tag++"num_batches_tracked",.buffer,"torch.int64",[],true⟩] := by rfl

theorem residual_gate_has_its_explicit_float32_dtype (dtype tag : String) (channels bottleneck kernel : ℕ) :
    (residual dtype tag channels bottleneck kernel).getLast?=
      some (parameter "torch.float32" (tag++"gate") []) := by rfl

theorem branch_keeps_both_convolutions_and_all_five_BN_entries
    (dtype : String) (index spatial perScale kernel : ℕ) :
    (branch dtype index spatial perScale kernel).length=7 := by rfl

theorem head_tables_keep_every_actual_parameter_array (dtype : String) (width rank hidden : ℕ) :
    (head dtype .quadraticUnfold width rank hidden).length=5 ∧
    (head dtype .mlpUnfold width rank hidden).length=5 ∧
    (head dtype .fullUnfold width rank hidden).length=2 := by exact ⟨rfl,rfl,rfl⟩

theorem prefix_changes_only_the_actual_registered_name (tag : String) (slot : Slot) :
    (prefixSlot tag slot).name=tag++slot.name ∧ (prefixSlot tag slot).role=slot.role ∧
      (prefixSlot tag slot).dtype=slot.dtype ∧ (prefixSlot tag slot).shape=slot.shape ∧
      (prefixSlot tag slot).legacyCounter=slot.legacyCounter := by exact ⟨rfl,rfl,rfl,rfl,rfl⟩

theorem wrapping_keeps_every_core_registration_and_adds_both_adapters
    (dtype : String) (plan : Plan) (config : Wrapper) :
    (wrapped dtype plan config).length=(plain dtype plan).length+14 := by
  simp [wrapped,residual]

theorem plain_branch_has_no_wrapper_prefix (dtype : String) (plan : Plan) :
    registry dtype plan none=plain dtype plan := rfl

end
end N2P3.NeuralRegisteredState
