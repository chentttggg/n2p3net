import N2P3.PersonalERPNineAdmissionWitness

namespace N2P3.PersonalERPNineInputWitness
open N2P3.FrozenState N2P3.StateLedger N2P3.ERPState
open N2P3.PersonalERPData N2P3.PersonalERPScalar N2P3.PersonalERPModelRuntime
open N2P3.ERPTemporalNPZ N2P3.ERPTemporalArrayViews N2P3.PersonalERPArrayIngress

noncomputable section
set_option maxRecDepth 8192
set_option maxHeartbeats 500000
set_option pp.maxSteps 2000
attribute [local instance] Classical.propDecidable
attribute [local simp] Except.bind Except.pure Except.map Except.mapError
attribute [local simp] Bind.bind Pure.pure Functor.map SeqRight.seqRight SeqLeft.seqLeft Seq.seq

def expectedContract : Fields := Fields.ofList
  [("channel_names",.sequence (encodeTexts ["FZ"])),
   ("channel_positions_m",.sequence (valuesOfList [.sequence (valuesOfList [.number 0,.number 0,.number (1/10)])])),
   ("preprocessing",.object (preprocessingRecord resolvedPreprocessing)),("source_reference",.text "left earlobe"),
   ("n_channels",.integer 1),("n_times",.integer 2),("simulation_status",.text "declared_mock"),
   ("ocular_artifact_processing",.object (Fields.ofList [("status",.text "not_applied")])),
   ("continuous_artifact_processing",.object (Fields.ofList [("status",.text "not_applied")]))]

def expectedAuthority : String := "n2p3-origin://BrainSync-GTN/participant/P001"

def expectedIdentityRows : List PersonalERPInput.IdentityRecord :=
  [⟨"P001",[expectedAuthority],[],"source_verified"⟩]

theorem decoded_position_cells_are_the_declared_cells (numeric : raw.positions.kind.numeric=true) :
    (PersonalERPGuardedViews.numericNative raw.positions numeric).values=[.real 0,.real 0,.real (1/10)] := by
  have injective : Function.Injective Cell.atom := by
    intro a b same
    cases same
    rfl
  apply List.map_injective_iff.2 injective
  exact (PersonalERPGuardedViews.numericNative raw.positions numeric).decoding

theorem guarded_coordinates_encode_the_actual_position_cells (C : ℕ)
    (shape : raw.positions.shape=[C,3]) (numeric : raw.positions.kind.numeric=true) :
    PersonalERPInput.coordinateJSON
      (PersonalERPGuardedViews.matrixOf (PersonalERPGuardedViews.numericNative raw.positions numeric) C 3 shape)
      (List.finRange C)=
      .ok [.sequence (valuesOfList [.number 0,.number 0,.number (1/10)])] := by
  have one : C=1 := by
    change [1,3]=[C,3] at shape
    exact (List.cons.inj shape).1.symm
  subst C
  dsimp only [PersonalERPGuardedViews.matrixOf,PersonalERPGuardedViews.matrix]
  simp [List.finRange,PersonalERPGuardedViews.matrix,decoded_position_cells_are_the_declared_cells,PersonalERPGuardedViews.flatIndex,
    PersonalERPArrayIngress.vector,PersonalERPInput.coordinateJSON,
    PersonalERPInput.atomJSONList,PersonalERPInput.atomJSON,List.ofFn_succ]

theorem input_contract_from_declared_fields (p : PersonalERPExecutionRuntime.Primitives) (h : Leaves p)
    (checked : Validated raw) :
    PersonalERPRawInputTail.inputContract (operations p) raw checked=.ok expectedContract := by
  have dimensions := every_guarded_raw_result_has_nine_rows checked
  have stripReference := h.strip "left earlobe" (by simp [literalStrings])
  have coordinates := guarded_coordinates_encode_the_actual_position_cells checked.tensor.C
    checked.positionShape checked.positionDType
  simp only [PersonalERPRawInputTail.inputContract,coordinates]
  simp [operations,dimensions.2.1,dimensions.2.2,
    PersonalERPModelRuntime.text,PersonalERPExecution.text,
    raw,texts,floats,fieldDefault,Fields.ofList,Fields.get,
    preprocessing_fields_are_known,stripReference,require,liftCheck,expectedContract]

end
end N2P3.PersonalERPNineInputWitness
