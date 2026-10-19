import N2P3.NeuralEvaluationControl

/-! Independent NC01-NC08 fixtures fixed before this slice's proofs. Small
slot tables exercise the load primitive, not an executable whole N2P3Net. -/
namespace N2P3.NeuralSavedControlWitnesses

open N2P3.NeuralConstructorConfig N2P3.NeuralRegisteredState N2P3.NeuralStrictLoad
open N2P3.NeuralEvaluationControl N2P3.PersonalState N2P3.FrozenState
open N2P3.StateLedger (numericScalar)
noncomputable section
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map

def smallArchitecture : Architecture :=
  ⟨1,3,1,1,4,[3],none,1,2,0,1,1,1,false,none⟩
def physical250 : Physical := ⟨1,8,250,-200⟩
def physical500 : Physical := ⟨1,8,500,-200⟩
def plan250 : Plan := ⟨modelArguments smallArchitecture physical250 .fullUnfold,2,2,featureClock smallArchitecture physical250⟩
def plan500 : Plan := ⟨modelArguments smallArchitecture physical500 .fullUnfold,2,2,featureClock smallArchitecture physical500⟩
def wrapper : Wrapper :=
  ⟨⟨1,1,1,3,.finite (1/4),.finite (1/4),0,0,.finite 0,.text "sample"⟩,rfl,rfl⟩

theorem same_persistent_registry_does_not_determine_the_physical_feature_clock :
    plain "torch.float32" plan250=plain "torch.float32" plan500 ∧
      (featureClock smallArchitecture physical250).sampleRate=125/2 ∧
      (featureClock smallArchitecture physical500).sampleRate=125 := by
  refine ⟨rfl,?_,?_⟩ <;> norm_num [featureClock,smallArchitecture,physical250,physical500]

theorem reconstructed_target_enable_flag_is_independent_of_tensor_values :
    (evalControl (constructorControl [] [] (some wrapper))).targetEnabled=some true ∧
      (evalControl (constructorControl [] [] none)).targetEnabled=none := by exact ⟨rfl,rfl⟩

theorem missing_weight_is_a_real_strict_failure :
    strictLoad preservingCast [weightSlot] []=.error (.missing "weight") := by
  simp [strictLoad,noUnexpected,loadEntries,loadOne,sourceForSlot,lookupTensor,weightSlot,parameter]

theorem unrelated_extra_model_key_is_a_real_strict_failure :
    strictLoad preservingCast [weightSlot] [("weight",finiteSavedTensor),("unexpected",finiteSavedTensor)]=
      .error NeuralStrictLoad.Failure.unexpected := by
  simp [strictLoad,noUnexpected,weightSlot,parameter]

theorem finite_saved_values_can_have_an_explicit_nonfinite_cast_observation :
    loadOne infinityCast weightSlot [("weight",finiteSavedTensor)]=
      .ok ⟨"torch.float32",[1],[.positiveInfinity]⟩ :=
  finite_saved_tensor_does_not_constrain_an_unspecified_cast_reply.1

def bnCounter : Slot := ⟨"bn.num_batches_tracked",.buffer,"torch.int64",[],true⟩

theorem omitted_ordinary_BN_counter_has_the_actual_legacy_success :
    strictLoad preservingCast [bnCounter] []=.ok [("bn.num_batches_tracked",⟨"torch.int64",[],[.finite 0]⟩)] := by
  simp [strictLoad,noUnexpected,loadEntries,loadOne,sourceForSlot,lookupTensor,bnCounter,zeroCounter,
    scalarCompatibility,NeuralStrictLoad.copyValues,numericScalar,preservingCast]

def meanTensor : Tensor := ⟨"torch.float32",[1,1,1],[.number 0]⟩
def negativeStd : Tensor := ⟨"torch.float32",[1,1,1],[.number (-1)]⟩

theorem separate_saved_scale_validation_cannot_be_replaced_by_weight_load :
    strictLoad preservingCast [weightSlot] [("weight",finiteSavedTensor)]=
      .ok [("weight",⟨"torch.float32",[1],[.finite 1]⟩)] ∧
    inputStatsGuard 1 [("input_mean",meanTensor),("input_std",negativeStd)]=false := by
  constructor
  · simp [strictLoad,noUnexpected,loadEntries,loadOne,sourceForSlot,lookupTensor,weightSlot,parameter,
      finiteSavedTensor,scalarCompatibility,NeuralStrictLoad.copyValues,numericScalar,preservingCast]
  · simp [inputStatsGuard,lookupTensor,validInputStats,meanTensor,negativeStd,positiveSaved,numericScalar]

theorem architecture_mismatch_stops_before_the_unrelated_later_statistics_gate :
    finish preservingCast (fun _ => true) plan250 "torch.float32" none []
      (.text "constructed") (.text "saved") [] []=.error NeuralEvaluationControl.Failure.architecture := by
  simp [finish]

end
end N2P3.NeuralSavedControlWitnesses
