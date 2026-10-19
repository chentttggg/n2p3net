import N2P3.PersonalERPFullAdmission

/-! Independent atomic failures exercise the source-derived order. No complete
reader, recipe or constructor result is supplied as an operation premise. -/
namespace N2P3.PersonalERPFullAdmissionOrder
open N2P3.FrozenState N2P3.StateLedger N2P3.PersonalERPAdmission
open N2P3.PersonalERPFullAdmission

noncomputable section
attribute [local instance] Classical.propDecidable

def deviceFault (p : Operations) : Operations :=
  { p with admission := { p.admission with original := { p.admission.original with
      finiteJSON := fun value => .ok value
      detachCPUClone := fun value => .ok value
      device := fun _ => .error (.sourceReject "declared device failure") } } }

theorem device_failure_before_nonobject_metadata (p : Operations) (state : TorchState) :
    constructERP (deviceFault p) .null state "invalid-device" =
      .error (.runtime (.inherited (.sourceReject "declared device failure"))) := by
  apply device_error_from_each_clone_precedes_metadata_get
    (deviceFault p) .null .null state "invalid-device" (.sourceReject "declared device failure")
  · rfl
  · intro entry _
    rfl
  · rfl

def numpyFault (p : Operations) : Operations :=
  { p with
    admission := { p.admission with
      original := { p.admission.original with
        finiteJSON := fun value => .ok value
        detachCPUClone := fun value => .ok value
        device := fun value => .ok value }
      selectedUniqueCount := fun _ => .ok 1
      selectedAnyNegative := fun _ => .ok false
      selectedAnyTooHigh := fun _ _ => .ok false }
    numpy := fun _ => .error .typeError }

theorem numpy_failure_before_nonobject_artifact (p : Operations) (tensor : TorchTensor) :
    constructERP (numpyFault p)
      (.object (noQualityRawMetadata "authority" (oneChannelContract .nil) .nil .null))
      (oneERPTensorState tensor) "cpu" =
      .error (.native "Tensor.numpy" "coefficient" .typeError) := by
  apply full_constructor_numpy_error_from_independent_atoms
    (numpyFault p) tensor .null .nil "cpu" "cpu" .typeError
  · simp [fieldKeys]
  · rfl
  · intro entry _
    rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl

def castFault (p : Operations) : Operations :=
  { p with
    asFloat64 := fun _ => .error .typeError
    copyFloat64 := fun _ => .error .valueError }

theorem cast_failure_before_copy_failure (p : Operations) (array : NativeArray) :
    castAll (castFault p) [("coefficient",array)] =
      .error (.native "np.asarray(float64)" "coefficient" .typeError) := by
  exact float64_failure_stops_before_copy_and_remaining_arrays
    (castFault p) "coefficient" array [] .typeError rfl

def copyFault (p : Operations) (array : N2P3.PersonalERPFitOutcome.Array64) : Operations :=
  { p with
    asFloat64 := fun _ => .ok array
    copyFloat64 := fun _ => .error .valueError }

theorem copy_failure_before_next_cast (p : Operations) (array later : NativeArray)
    (converted : N2P3.PersonalERPFitOutcome.Array64) :
    castAll (copyFault p converted) [("first",array),("later",later)] =
      .error (.native "ndarray.copy" "first" .valueError) := by
  exact array_copy_failure_stops_before_remaining_casts
    (copyFault p converted) "first" array converted [("later",later)] .valueError rfl rfl

theorem mapping_method_error (site : String) :
    getMapping .null site = .error (.native "get" site .attributeError) := rfl

theorem contract_explicit_type_guard_error :
    contractDictionary .null = .error (.runtime
      (.sourceValueError "calibrated model lacks participant and input contracts")) := rfl

theorem keyword_expansion_type_error (site : String) :
    kwargsMapping .null site = .error (.native "keyword expansion" site .typeError) := rfl

theorem string_subscription_type_error (site : String) :
    indexedMapping .null site = .error (.native "string-key subscription" site .typeError) := rfl

end
end N2P3.PersonalERPFullAdmissionOrder
