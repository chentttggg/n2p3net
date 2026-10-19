import N2P3.NeuralSavedStateProjection

namespace N2P3.NegativeNeuralSavedStateProjection
open N2P3.NeuralSavedStateProjection N2P3.PersonalState N2P3.FrozenState

example : stripPrefix "trunk." "xtrunk.weight" = some "weight" := by
  rw [fixed_nonprefix_is_not_a_selected_model_tensor]
  simp

example : stripPrefix "trunk." "trunk.trunk.weight" = some "weight" := by
  rw [fixed_repeated_prefix_is_removed_only_once]
  have different : (some "trunk.weight" : Option String) ≠ some "weight" := by decide
  simp only [different]

example : lookupTensor "input_mean" (project "trunk." witnessState) = some witnessTensor := by
  rw [fixed_saved_statistics_are_separate_from_strict_model_state.2.2.1]
  simp

example : (arguments ⟨.object .nil, project "trunk." witnessState⟩).inputStd =
    (arguments ⟨.object .nil, witnessState⟩).inputStd := by
  change lookupTensor "input_std" (project "trunk." witnessState) = lookupTensor "input_std" witnessState
  rw [fixed_saved_statistics_are_separate_from_strict_model_state.2.2.2]
  simp [witnessState, lookupTensor]

end N2P3.NegativeNeuralSavedStateProjection
