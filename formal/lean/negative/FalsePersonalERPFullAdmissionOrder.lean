import N2P3.PersonalERPFullAdmissionOrder

/-! The alternatives were listed in full_admission_connection_contract_03:
metadata access first, fused conversion, and interchangeable error classes. -/
namespace N2P3.NegativePersonalERPFullAdmissionOrder
open N2P3.FrozenState N2P3.PersonalERPAdmission
open N2P3.PersonalERPFullAdmission N2P3.PersonalERPFullAdmissionOrder
noncomputable section

example (p : Operations) (state : TorchState) :
    constructERP (deviceFault p) .null state "invalid-device" =
      .error (.native "get" "decision_score metadata" .attributeError) := by
  rw [device_failure_before_nonobject_metadata]
  try simp +decide

example (p : Operations) (tensor : TorchTensor) :
    constructERP (numpyFault p)
      (.object (noQualityRawMetadata "authority" (oneChannelContract .nil) .nil .null))
      (oneERPTensorState tensor) "cpu" =
      .error (.native "get" "ERP artifact" .attributeError) := by
  rw [numpy_failure_before_nonobject_artifact]
  simp +decide

example (p : Operations) (array : NativeArray) :
    castAll (castFault p) [("coefficient",array)] =
      .error (.native "ndarray.copy" "coefficient" .valueError) := by
  rw [cast_failure_before_copy_failure]
  simp +decide

example (p : Operations) (array later : NativeArray)
    (converted : N2P3.PersonalERPFitOutcome.Array64) :
    castAll (copyFault p converted) [("first",array),("later",later)] =
      .error (.native "np.asarray(float64)" "later" .typeError) := by
  rw [copy_failure_before_next_cast]
  simp +decide

example : getMapping .null "metadata" = .error (.native "get" "metadata" .typeError) := by
  rw [mapping_method_error]
  try simp +decide

example : contractDictionary .null = .error (.native "get" "input contract" .attributeError) := by
  rw [contract_explicit_type_guard_error]
  try simp +decide

example : kwargsMapping .null "recipe" = .error (.native "keyword expansion" "recipe" .attributeError) := by
  rw [keyword_expansion_type_error]
  try simp +decide

example : indexedMapping .null "baseline" = .error (.native "string-key subscription" "baseline" .attributeError) := by
  rw [string_subscription_type_error]
  try simp +decide

end
end N2P3.NegativePersonalERPFullAdmissionOrder
