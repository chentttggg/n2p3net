import N2P3.NeuralSavedControlWitnesses

namespace N2P3.NegativeNeuralSavedControl
open N2P3.NeuralConstructorConfig N2P3.NeuralRegisteredState N2P3.NeuralStrictLoad
open N2P3.NeuralEvaluationControl N2P3.NeuralSavedControlWitnesses

-- NC01: identical saved parameter shapes do not fix the physical feature clock.
example : (featureClock smallArchitecture physical250).sampleRate=125 := by
  rw [same_persistent_registry_does_not_determine_the_physical_feature_clock.2.1]
  norm_num

-- NC02: a tensor-only image cannot replace the actual enabled target flag.
example : (evalControl (constructorControl [] [] (some wrapper))).targetEnabled=some false := by
  rw [reconstructed_target_enable_flag_is_independent_of_tensor_values.1]
  simp

-- NC03: strict load cannot silently drop a missing ordinary weight.
example : strictLoad preservingCast [weightSlot] []=.ok [] := by
  rw [missing_weight_is_a_real_strict_failure]
  simp

-- NC04: an unrelated projected model key is not silently admitted.
example : strictLoad preservingCast [weightSlot] [("weight",finiteSavedTensor),("unexpected",finiteSavedTensor)]=.ok [] := by
  rw [unrelated_extra_model_key_is_a_real_strict_failure]
  simp

-- NC05: finite source scalars do not supply cast-value/finiteness preservation.
example : loadOne infinityCast weightSlot [("weight",finiteSavedTensor)]=
    .ok ⟨"torch.float32",[1],[.finite 1]⟩ := by
  rw [finite_saved_values_can_have_an_explicit_nonfinite_cast_observation]
  simp

-- NC06: version-None ordinary BN inserts its legacy missing counter.
example : strictLoad preservingCast [bnCounter] []=.error (.missing "bn.num_batches_tracked") := by
  rw [omitted_ordinary_BN_counter_has_the_actual_legacy_success]
  simp

-- NC07: even successful strict parameter loading does not validate input std.
example : inputStatsGuard 1 [("input_mean",meanTensor),("input_std",negativeStd)]=true := by
  rw [separate_saved_scale_validation_cannot_be_replaced_by_weight_load.2]
  simp

-- NC08: architecture equality is checked before the later statistics guard.
example : finish preservingCast (fun _ => true) plan250 "torch.float32" none []
    (.text "constructed") (.text "saved") [] []=.error NeuralEvaluationControl.Failure.inputStatistics := by
  rw [architecture_mismatch_stops_before_the_unrelated_later_statistics_gate]
  simp

end N2P3.NegativeNeuralSavedControl
